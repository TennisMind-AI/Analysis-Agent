# video-analyzer deployment starter

This is a minimal backend wrapper for the upstream `video-analyzer` project.

## What it does

- Accepts a video upload from the frontend
- Rejects videos longer than 5 seconds when `ffprobe` is available
- Calls the `video-analyzer` CLI
- Returns the generated `analysis.json`

## Run with Docker (recommended if IDEA has no Python environment)

This is the easiest way to use FastAPI without configuring Python inside IDEA first.

1. Install Docker Desktop
2. In this project directory, start the API:

```bash
docker compose up --build
```

3. If you want cloud inference, set environment variables before startup:

```bash
cp .env.example .env
# edit .env and fill in VIDEO_ANALYZER_API_KEY
docker compose up --build
```

4. Test:

```bash
curl http://127.0.0.1:8000/health
```

## Run locally

1. Install system dependency:

```bash
brew install ffmpeg
```

2. Install the upstream analyzer package in the same Python environment:

```bash
git clone https://github.com/byjlw/video-analyzer.git
cd video-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cd ..
```

3. Install this API wrapper:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

4. Optional environment variables for cloud inference:

```bash
export VIDEO_ANALYZER_CLIENT=openai_api
export VIDEO_ANALYZER_API_KEY=your-key
export VIDEO_ANALYZER_API_URL=https://openrouter.ai/api/v1
export VIDEO_ANALYZER_MODEL=meta-llama/llama-3.2-11b-vision-instruct:free
```

## API

### `GET /health`

Health check.

### `POST /analyze`

Primary frontend endpoint.

Request:

```json
{
  "user_id": "user-123",
  "video_url": "https://example.com/demo.mp4"
}
```

Behavior:

- Downloads the remote video
- Runs video analysis
- Fetches user context from context agent by `user_id`
- Separates user memory by `user_id`
- Sends the analysis JSON into trigger logic
- Saves updated context back to context agent
- Returns the final text response for the frontend

Response:

```json
{
  "text": "coaching feedback for TTS...",
  "analysis": {},
  "metadata": {}
}
```

The frontend should treat `text` as the primary response field.
It is designed to be directly readable by a voice or TTS system.

Example:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user-123","video_url":"https://example.com/demo.mp4"}'
```

### `POST /trigger`

Accepts analysis JSON and produces the user-facing text response.

Example:

```bash
curl -X POST http://127.0.0.1:8000/trigger \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user-123","analysis":{"sport":"tennis","summary":"Wall practice"}}'
```

## Optional Context Agent Integration

If you already have a separate context agent service, configure:

```bash
CONTEXT_AGENT_LOGS_URL=https://your-service/context/logs
CONTEXT_AGENT_TRIGGER_URL=https://your-service/context/trigger
```

Expected logs request:

```http
GET /logs?user_id=user-123
```

Expected get response body:

```json
{
  "context": {
    "preferred_focus": "contact point",
    "recurring_issue": "late preparation",
    "dominant_hand": "right"
  }
}
```

Expected trigger request body:

```json
{
  "user_id": "user-123",
  "context": {},
  "analysis": {},
  "latest_text": "..."
}
```

### `POST /analyze-upload`

Multipart upload with a `file` field containing the video.

Example:

```bash
curl -X POST \
  -F "file=@demo.mp4" \
  http://127.0.0.1:8000/analyze-upload
```

## Notes

- The Dockerfile only installs this wrapper and `ffmpeg`.
- You still need to install `video-analyzer` in the image or bake the upstream repo into a custom image.
- For hackathon speed, using an OpenAI-compatible API is usually simpler than serving local Ollama + vision model on your own GPU.
- If you want a fully containerized setup, the next step is to copy or mount the upstream `video-analyzer` package into the image and install it during `docker build`.
