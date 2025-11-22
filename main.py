import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from duckduckgo_search import DDGS
from imap_tools import MailBox, AND
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "gpt-oss:20b")
DUCKDUCKGO_SEARCH_ENABLED = os.getenv("DUCKDUCKGO_SEARCH_ENABLED", "True").lower() == "true"
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
class PromptRequest(BaseModel):
    prompt: str

class OllamaResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool

# --- Tool Functions ---

def search_duckduckgo(query: str):
    """Searches DuckDuckGo for the given query and returns the results."""
    if not DUCKDUCKGO_SEARCH_ENABLED:
        return "DuckDuckGo search is disabled."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            return json.dumps(results)
    except Exception as e:
        return f"Error searching DuckDuckGo: {e}"

def read_emails(mark_seen: bool = False):
    """Reads unread emails from the inbox."""
    if not EMAIL_ENABLED:
        return "Email access is disabled."
    if not all([IMAP_SERVER, IMAP_USERNAME, IMAP_PASSWORD]):
        return "IMAP credentials are not configured."
    try:
        with MailBox(IMAP_SERVER).login(IMAP_USERNAME, IMAP_PASSWORD, 'INBOX') as mailbox:
            emails = []
            for msg in mailbox.fetch(AND(seen=False)):
                emails.append({
                    "from": msg.from_,
                    "subject": msg.subject,
                    "date": msg.date.isoformat(),
                    "text": msg.text
                })
                if mark_seen:
                    mailbox.seen(msg.uid, True)
            return json.dumps(emails)
    except Exception as e:
        return f"Error reading emails: {e}"

# Dictionary of available tools
TOOLS = {
    "search_duckduckgo": search_duckduckgo,
    "read_emails": read_emails,
}

# --- Ollama Interaction ---

def get_ollama_response(prompt: str):
    """Sends a prompt to the Ollama model and gets a response."""
    full_url = f"{OLLAMA_BASE_URL}/api/generate"
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False  # We'll handle the full response
    }
    try:
        response = requests.post(full_url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Ollama: {e}")

# --- API Endpoints ---

@app.post("/prompt")
def process_prompt(request: PromptRequest):
    """
    Receives a prompt, determines if a tool should be used,
    and returns the LLM's response.
    """
    prompt = request.prompt
    
    # Simple tool detection (can be improved with more sophisticated logic)
    if prompt.startswith("!search"):
        query = prompt.replace("!search", "").strip()
        tool_result = search_duckduckgo(query)
        # Augment the prompt with the tool's result
        augmented_prompt = f"Based on the following search results:\n{tool_result}\n\nAnswer the user's original query: {query}"
        ollama_response = get_ollama_response(augmented_prompt)
        return {"response": ollama_response["response"]}

    elif prompt.startswith("!emails"):
        tool_result = read_emails()
        # Augment the prompt with the tool's result
        augmented_prompt = f"Here are the latest unread emails:\n{tool_result}\n\nSummarize them or take action as requested."
        ollama_response = get_ollama_response(augmented_prompt)
        return {"response": ollama_response["response"]}

    else:
        # No tool detected, send prompt directly to Ollama
        ollama_response = get_ollama_response(prompt)
        return {"response": ollama_response["response"]}

@app.get("/")
def read_root():
    return {"message": "MCP-Local-LLM server is running."}
