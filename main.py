import json
import os
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
except Exception:  # pragma: no cover - optional dependency
    Credentials = None
    build = None

# Load environment variables from .env file
load_dotenv()

# Core configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "gpt-oss:20b")

# Search / browse
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
DUCKDUCKGO_SEARCH_ENABLED = os.getenv("DUCKDUCKGO_SEARCH_ENABLED", "True").lower() == "true"
PLAYWRIGHT_BROWSER = os.getenv("PLAYWRIGHT_BROWSER", "chromium")

# RAG / Weaviate
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_INDEX = os.getenv("WEAVIATE_INDEX", "Documents")

# Calendar
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
GOOGLE_TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", "./google_tokens.json")
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")

# Local access
FILE_ROOT = Path(os.getenv("FILE_ROOT", ".")).resolve()
SHELL_ALLOWLIST = [cmd.strip() for cmd in os.getenv("SHELL_ALLOWLIST", "").split(",") if cmd.strip()]
SHELL_DENYLIST = [cmd.strip() for cmd in os.getenv("SHELL_DENYLIST", "").split(",") if cmd.strip()]

# Email (kept for compatibility)
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "False").lower() == "true"
IMAP_SERVER = os.getenv("IMAP_SERVER")
IMAP_USERNAME = os.getenv("IMAP_USERNAME")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD")

# Initialize FastAPI app
app = FastAPI()

# Allow browser clients (e.g., the frontend on port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request and response
class Message(BaseModel):
    role: str
    content: str


class PromptRequest(BaseModel):
    prompt: str
    messages: Optional[List[Message]] = None


class OllamaResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool


# --- Helper functions ---
def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate an embedding via Ollama; returns None if unavailable."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": MODEL, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("embedding")
    except Exception:
        return None


def ensure_within_root(path_str: str) -> Path:
    """Ensure the path stays within FILE_ROOT."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = FILE_ROOT / path
    path = path.resolve()
    if FILE_ROOT not in path.parents and path != FILE_ROOT:
        raise ValueError(f"Path {path} is outside the allowed root {FILE_ROOT}")
    return path


def format_search_results(results: List[dict]) -> str:
    lines = []
    for idx, item in enumerate(results, start=1):
        title = item.get("title") or item.get("name") or "Untitled"
        url = item.get("url") or item.get("href") or item.get("link") or ""
        snippet = item.get("description") or item.get("body") or item.get("snippet") or ""
        lines.append(f"{idx}. {title}\n   {url}\n   {snippet}")
    return "\n".join(lines) if lines else "No results."


# --- Tool Functions ---
def search_brave(query: str) -> str:
    """Search Brave; fall back to DuckDuckGo if not configured."""
    headers = {
        "Accept": "application/json",
    }
    if BRAVE_API_KEY:
        headers["X-Subscription-Token"] = BRAVE_API_KEY
    else:
        return f"Brave API key missing. Falling back.\n{search_duckduckgo(query)}"

    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": 5},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("web", {}).get("results", [])
        return format_search_results(results)
    except Exception as exc:
        return f"Brave search failed ({exc}); fallback results:\n{search_duckduckgo(query)}"


def search_duckduckgo(query: str) -> str:
    """Search DuckDuckGo for the given query."""
    if not DUCKDUCKGO_SEARCH_ENABLED:
        return "DuckDuckGo search is disabled."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            return format_search_results(results)
    except Exception as e:
        return f"Error searching DuckDuckGo: {e}"


def browse_with_playwright(url: str) -> str:
    """Use Playwright to fetch a rendered page and extract readable text."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return "Playwright is not installed. Run `pip install playwright` and `playwright install`."

    text_content = ""
    try:
        with sync_playwright() as p:
            browser_type = getattr(p, PLAYWRIGHT_BROWSER, None)
            if browser_type is None:
                return f"Invalid PLAYWRIGHT_BROWSER '{PLAYWRIGHT_BROWSER}'. Use chromium, firefox, or webkit."
            browser = browser_type.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=20000)
            html = page.content()
            browser.close()
            soup = BeautifulSoup(html, "html.parser")
            [s.extract() for s in soup(["script", "style"])]
            text_content = " ".join(soup.get_text().split())
    except Exception as exc:
        return f"Error rendering page: {exc}"

    return text_content[:8000] if text_content else "No readable content extracted."


def ensure_weaviate_schema():
    """Ensure the class exists in Weaviate."""
    try:
        resp = requests.get(f"{WEAVIATE_URL}/v1/schema", timeout=10)
        resp.raise_for_status()
        schema = resp.json()
        classes = [c.get("class") for c in schema.get("classes", [])]
        if WEAVIATE_INDEX in classes:
            return

        new_class = {
            "class": WEAVIATE_INDEX,
            "vectorizer": "none",
            "properties": [
                {"name": "title", "dataType": ["text"]},
                {"name": "text", "dataType": ["text"]},
                {"name": "source", "dataType": ["text"]},
            ],
        }
        create_resp = requests.post(
            f"{WEAVIATE_URL}/v1/schema",
            json=new_class,
            timeout=10,
        )
        create_resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Weaviate schema error: {exc}")


def ingest_to_weaviate(chunks: List[str], source: str) -> int:
    """Push chunked text into Weaviate with externally supplied vectors."""
    ensure_weaviate_schema()
    created = 0
    for idx, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        if not embedding:
            continue
        payload = {
            "class": WEAVIATE_INDEX,
            "properties": {
                "title": f"{source} #{idx + 1}",
                "text": chunk,
                "source": source,
            },
            "vector": embedding,
        }
        try:
            resp = requests.post(f"{WEAVIATE_URL}/v1/objects", json=payload, timeout=15)
            resp.raise_for_status()
            created += 1
        except Exception:
            continue
    return created


def query_weaviate(question: str, limit: int = 5) -> str:
    """Query Weaviate using a nearVector search."""
    ensure_weaviate_schema()
    embedding = generate_embedding(question)
    if not embedding:
        return "Embedding generation failed; ensure your Ollama model supports embeddings."

    graphql = {
        "query": f"""
        {{
          Get {{
            {WEAVIATE_INDEX}(nearVector: {{vector: {json.dumps(embedding)}}}, limit: {limit}) {{
              title
              text
              source
            }}
          }}
        }}
        """
    }
    try:
        resp = requests.post(f"{WEAVIATE_URL}/v1/graphql", json=graphql, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", {}).get("Get", {}).get(WEAVIATE_INDEX, [])
        if not data:
            return "No RAG results found."
        lines = []
        for item in data:
            lines.append(f"- {item.get('title')}: {item.get('text')[:400]} (source: {item.get('source')})")
        return "\n".join(lines)
    except Exception as exc:
        return f"Weaviate query failed: {exc}"


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> List[str]:
    """Simple text chunker."""
    chunks = []
    start = 0
    total_len = len(text)
    while start < total_len:
        end = min(total_len, start + size)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= total_len:
            break
        start = end - overlap
    return [c for c in chunks if c]


def read_local_file(path: str, max_bytes: int = 40000) -> str:
    try:
        target = ensure_within_root(path)
        if not target.exists():
            return f"File not found: {target}"
        data = target.read_text(encoding="utf-8", errors="ignore")
        if len(data) > max_bytes:
            return data[:max_bytes] + "\n\n...[truncated]..."
        return data
    except Exception as exc:
        return f"Error reading file: {exc}"


def list_directory(path: str = ".") -> str:
    try:
        target = ensure_within_root(path)
        if target.is_file():
            return f"{target} is a file."
        entries = sorted([p.name for p in target.iterdir()])
        return "\n".join(entries)
    except Exception as exc:
        return f"Error listing directory: {exc}"


def run_shell(command: str) -> str:
    stripped = command.strip()
    if any(stripped.startswith(block) for block in SHELL_DENYLIST):
        return f"Command blocked by denylist: {stripped}"
    if SHELL_ALLOWLIST and not any(stripped.startswith(allow) for allow in SHELL_ALLOWLIST):
        return f"Command not in allowlist: {stripped}"

    try:
        result = subprocess.run(
            stripped,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout.strip()
        err = result.stderr.strip()
        return (output + ("\n" + err if err else "")).strip() or "(no output)"
    except Exception as exc:
        return f"Shell error: {exc}"


def list_calendar_events(max_results: int = 5) -> str:
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and GOOGLE_REDIRECT_URI):
        return "Google Calendar is not configured (missing client ID/secret/redirect)."
    if Credentials is None or build is None:
        return "Google API libraries not installed. Install google-auth, google-auth-oauthlib, google-api-python-client."
    if not os.path.exists(GOOGLE_TOKEN_PATH):
        return f"Token file not found at {GOOGLE_TOKEN_PATH}. Complete OAuth flow first."

    try:
        creds = Credentials.from_authorized_user_file(
            GOOGLE_TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/calendar.readonly", "https://www.googleapis.com/auth/calendar.events"],
        )
        service = build("calendar", "v3", credentials=creds)
        now = datetime.utcnow().isoformat() + "Z"
        events_result = service.events().list(
            calendarId=GOOGLE_CALENDAR_ID,
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = events_result.get("items", [])
        if not events:
            return "No upcoming events found."

        lines = []
        for event in events:
            start = event.get("start", {}).get("dateTime") or event.get("start", {}).get("date")
            summary = event.get("summary", "No title")
            lines.append(f"- {start}: {summary}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Calendar error: {exc}"


def read_emails(mark_seen: bool = False) -> str:
    """Reads unread emails from the inbox."""
    if not EMAIL_ENABLED:
        return "Email access is disabled."
    if not all([IMAP_SERVER, IMAP_USERNAME, IMAP_PASSWORD]):
        return "IMAP credentials are not configured."
    try:
        from imap_tools import AND, MailBox  # lazy import

        with MailBox(IMAP_SERVER).login(IMAP_USERNAME, IMAP_PASSWORD, "INBOX") as mailbox:
            emails = []
            for msg in mailbox.fetch(AND(seen=False)):
                emails.append(
                    {
                        "from": msg.from_,
                        "subject": msg.subject,
                        "date": msg.date.isoformat(),
                        "text": msg.text,
                    }
                )
                if mark_seen:
                    mailbox.seen(msg.uid, True)
            return json.dumps(emails)
    except Exception as e:
        return f"Error reading emails: {e}"


# --- Ollama Interaction ---
def render_history(history: List[Message], max_messages: int = 10, max_chars: int = 8000) -> str:
    """Format a subset of history for grounding."""
    trimmed = history[-max_messages:] if history else []
    lines = []
    total = 0
    for msg in trimmed:
        piece = f"{msg.role.capitalize()}: {msg.content.strip()}"
        total += len(piece)
        if total > max_chars:
            break
        lines.append(piece)
    return "\n".join(lines)


def get_ollama_response(prompt: str, history: Optional[List[Message]] = None):
    """Sends a prompt + optional history to the Ollama model and gets a response."""
    history_text = render_history(history or [])
    prompt_body = f"{history_text}\n\nUser: {prompt}\nAssistant:" if history_text else prompt
    full_url = f"{OLLAMA_BASE_URL}/api/generate"
    try:
        response = requests.post(full_url, json={"model": MODEL, "prompt": prompt_body, "stream": False}, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Ollama: {e}")


# --- API Endpoints ---
@app.post("/prompt")
def process_prompt(request: PromptRequest):
    """
    Receives a prompt, routes to tools when detected (automatic heuristics),
    and returns the LLM's response.
    Supported commands (explicit):
    !search, !browse, !rag_ingest, !rag, !calendar, !read, !ls, !shell
    """
    prompt = request.prompt.strip()
    history = request.messages or []

    tool, payload = detect_tool(prompt)
    if tool is None:
        tool, payload = llm_suggest_tool(prompt, history=history)

    if tool == "search":
        query = payload or prompt
        results = search_brave(query)
        augmented_prompt = f"Conversation so far:\n{render_history(history)}\n\nSearch results:\n{results}\n\nAnswer the question: {query}"
        ollama_response = get_ollama_response(augmented_prompt, history=history)
        return {"response": ollama_response.get("response", "")}

    if tool == "browse":
        url = payload["url"]
        question = payload.get("question") or "Summarize the page."
        page_text = browse_with_playwright(url)
        augmented_prompt = f"Conversation so far:\n{render_history(history)}\n\nPage content from {url}:\n{page_text}\n\nTask: {question}"
        ollama_response = get_ollama_response(augmented_prompt, history=history)
        return {"response": ollama_response.get("response", "")}

    if tool == "rag_ingest":
        path = payload
        if not path:
            return {"response": "Usage: !rag_ingest <path-to-file>"}
        file_text = read_local_file(path)
        if file_text.startswith("Error") or file_text.startswith("File not found"):
            return {"response": file_text}
        chunks = chunk_text(file_text)
        created = ingest_to_weaviate(chunks, source=path)
        return {"response": f"Ingested {created} chunks into Weaviate for {path}."}

    if tool == "rag_query":
        question = payload
        if not question:
            return {"response": "Usage: !rag <question>"}
        context = query_weaviate(question)
        augmented_prompt = f"Conversation so far:\n{render_history(history)}\n\nUse the retrieved context to answer.\nContext:\n{context}\n\nQuestion: {question}"
        ollama_response = get_ollama_response(augmented_prompt, history=history)
        return {"response": ollama_response.get("response", "")}

    if tool == "calendar":
        events = list_calendar_events()
        return {"response": events}

    if tool == "read":
        if not payload:
            return {"response": "Usage: !read <path>"}
        return {"response": read_local_file(payload)}

    if tool == "ls":
        return {"response": list_directory(payload or ".")}

    if tool == "shell":
        if not payload:
            return {"response": "Usage: !shell <command>"}
        return {"response": run_shell(payload)}

    # No tool detected, send prompt directly to Ollama
    ollama_response = get_ollama_response(prompt, history=history)
    return {"response": ollama_response.get("response", "")}


@app.get("/")
def read_root():
    return {"message": "MCP-Local-LLM server is running."}


# --- Intent detection for tools ---

URL_REGEX = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def extract_url(text: str) -> Optional[str]:
    match = URL_REGEX.search(text)
    if match:
        return match.group(0)
    return None


def detect_tool(prompt: str):
    """
    Lightweight heuristics so users don't need explicit bang commands.
    Returns (tool_name, payload) or (None, None) to fall back to direct LLM.
    """
    lower = prompt.lower()

    # Explicit bang commands (still supported)
    if prompt.startswith("!search"):
        return ("search", prompt.replace("!search", "", 1).strip())
    if prompt.startswith("!browse"):
        parts = prompt.split(maxsplit=2)
        if len(parts) >= 2:
            url = parts[1]
            question = parts[2] if len(parts) > 2 else "Summarize the page."
            return ("browse", {"url": url, "question": question})
    if prompt.startswith("!rag_ingest"):
        return ("rag_ingest", prompt.replace("!rag_ingest", "", 1).strip())
    if prompt.startswith("!rag"):
        return ("rag_query", prompt.replace("!rag", "", 1).strip())
    if prompt.startswith("!calendar"):
        return ("calendar", None)
    if prompt.startswith("!read"):
        return ("read", prompt.replace("!read", "", 1).strip())
    if prompt.startswith("!ls"):
        return ("ls", prompt.replace("!ls", "", 1).strip() or ".")
    if prompt.startswith("!shell"):
        return ("shell", prompt.replace("!shell", "", 1).strip())

    # Heuristics
    url = extract_url(prompt)
    if url:
        return ("browse", {"url": url, "question": prompt})

    if any(kw in lower for kw in ["search for", "look up", "find info", "what is", "who is"]):
        return ("search", prompt)

    if any(kw in lower for kw in ["news", "headline", "headlines", "breaking", "top stories", "latest news"]):
        return ("search", prompt)

    if "calendar" in lower or "schedule" in lower:
        return ("calendar", None)

    if any(kw in lower for kw in ["list files", "ls ", "show directory", "show files"]):
        return ("ls", ".")

    if any(kw in lower for kw in ["read file", "open file", "show file", "view file"]):
        # Try to extract a path between quotes/backticks or after keywords
        path_match = re.search(r"(?:file|path)\s+[`'\"]?([^`'\"\\s]+)", prompt, re.IGNORECASE)
        if path_match:
            return ("read", path_match.group(1))

    # Allow explicit "run: <cmd>" phrasing for shell
    run_match = re.search(r"(?:run|execute|shell)[:\s]+(`?)([^`]+)\1", prompt, re.IGNORECASE)
    if run_match:
        return ("shell", run_match.group(2).strip())

    return (None, None)


def llm_suggest_tool(prompt: str, history: Optional[List[Message]] = None):
    """
    Ask the LLM to pick the best tool when heuristics do not trigger.
    Returns (tool, payload) or (None, None) on failure.
    """
    instruction = """You are a router. Choose the best tool for the user's request.
Tools:
- search: Web search for general info.
- browse: Load and summarize a specific URL (include URL in input).
- rag_query: Retrieve from vector store for knowledge recall.
- calendar: Show upcoming events.
- read: Read a file (include path).
- ls: List directory contents (include path or .).
- shell: Run a safe shell command (respect allowlist).
Respond ONLY with a JSON object: {"tool":"<name or none>","input":"<payload or empty>"}.
If no tool is needed, return {"tool":"none","input":""}.
"""
    hist_text = render_history(history or [])
    composed = f"{instruction}\nConversation:\n{hist_text}\nUser request: {prompt}\nYour JSON:"
    try:
        resp = get_ollama_response(composed, history=history)
        text = resp.get("response", "")
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(text[start:end+1])
            tool = parsed.get("tool")
            payload = parsed.get("input")
            if tool and tool != "none":
                return (tool, payload)
    except Exception:
        pass
    return (None, None)
