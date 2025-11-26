# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser (Chromium) with required dependencies
RUN python -m playwright install --with-deps chromium

# Copy the rest of the application's code into the container at /app
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define environment variables
ENV OLLAMA_BASE_URL="http://host.docker.internal:11434"
ENV MODEL="gpt-oss:20b"
ENV DUCKDUCKGO_SEARCH_ENABLED="True"
ENV EMAIL_ENABLED="False"
ENV IMAP_SERVER=""
ENV IMAP_USERNAME=""
ENV IMAP_PASSWORD=""

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
