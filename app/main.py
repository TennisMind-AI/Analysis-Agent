import json
import os
import shutil
import subprocess
import tempfile
import base64
import time
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
import requests


APP_NAME = "video-analyzer-service"
MAX_VIDEO_SECONDS = 15
MAX_UPLOAD_SIZE_BYTES = 30 * 1024 * 1024
ANALYZER_OUTPUT_NAME = "analysis.json"
MEMORY_DIR = Path("/app/data/memory")
USER_AGENT = "video-analyzer-service/1.0"


app = FastAPI(title=APP_NAME)


class AnalyzeVideoRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    video_url: str = Field(..., min_length=1)


class TriggerRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    analysis: dict[str, Any]


def _safe_user_id(user_id: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in user_id)
    return sanitized[:120] or "anonymous"


def _memory_path(user_id: str) -> Path:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return MEMORY_DIR / f"{_safe_user_id(user_id)}.json"


def _load_memory(user_id: str) -> dict[str, Any]:
    path = _memory_path(user_id)
    if not path.exists():
        return {"user_id": user_id, "history": []}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"user_id": user_id, "history": []}


def _save_memory(user_id: str, memory: dict[str, Any]) -> None:
    path = _memory_path(user_id)
    path.write_text(json.dumps(memory, ensure_ascii=True, indent=2))


def _fetch_context_agent_context(user_id: str) -> dict[str, Any] | None:
    context_logs_url = os.getenv("CONTEXT_AGENT_LOGS_URL") or os.getenv("CONTEXT_AGENT_GET_URL")
    if not context_logs_url:
        return None

    response = requests.get(
        context_logs_url,
        params={"user_id": user_id},
        timeout=30,
    )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Context agent logs fetch failed",
                "status_code": response.status_code,
                "body": response.text[-4000:],
            },
        )

    data = response.json()
    if isinstance(data, dict) and "context" in data and isinstance(data["context"], dict):
        return data["context"]
    if isinstance(data, dict):
        return data
    return None


def _log_to_context_agent(
    user_id: str,
    context: dict[str, Any],
    analysis: dict[str, Any],
    text: str,
) -> None:
    context_trigger_url = os.getenv("CONTEXT_AGENT_TRIGGER_URL")
    if not context_trigger_url:
        return

    raw_confidence = analysis.get("confidence", 0.0)
    if isinstance(raw_confidence, (int, float)):
        confidence_value = float(raw_confidence)
    elif isinstance(raw_confidence, str):
        normalized = raw_confidence.strip().lower()
        confidence_map = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.9,
            "very high": 0.95,
            "very low": 0.15,
            "uncertain": 0.2,
        }
        confidence_value = confidence_map.get(normalized, 0.0)
    else:
        confidence_value = 0.0

    response = requests.post(
        context_trigger_url,
        json={
            "user_id": user_id,
            "event_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "analysis_feedback",
            "payload": {
                "value": confidence_value,
                "context": (
                    f"user_id={user_id}; "
                    f"summary={analysis.get('summary', '')}; "
                    f"stroke_type={analysis.get('stroke_type', '')}; "
                    f"issues={', '.join(analysis.get('issues') or [])}; "
                    f"coaching_tips={', '.join(analysis.get('coaching_tips') or [])}; "
                    f"preferred_focus={context.get('preferred_focus', '')}; "
                    f"recurring_issue={context.get('recurring_issue', '')}; "
                    f"latest_text={text}"
                ),
            },
        },
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Context agent trigger failed",
                "status_code": response.status_code,
                "body": response.text[-4000:],
            },
        )


def _ensure_video_suffix(filename: str) -> None:
    allowed = {".mp4", ".mov", ".m4v", ".webm", ".avi"}
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")


def _run_ffprobe(video_path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def _build_analyzer_command(video_path: Path, output_dir: Path) -> list[str]:
    command = [
        "video-analyzer",
        str(video_path),
        "--output-dir",
        str(output_dir),
    ]

    client = os.getenv("VIDEO_ANALYZER_CLIENT")
    api_key = os.getenv("VIDEO_ANALYZER_API_KEY")
    api_url = os.getenv("VIDEO_ANALYZER_API_URL")
    model = os.getenv("VIDEO_ANALYZER_MODEL")

    if client:
        command.extend(["--client", client])
    if api_key:
        command.extend(["--api-key", api_key])
    if api_url:
        command.extend(["--api-url", api_url])
    if model:
        command.extend(["--model", model])

    return command


def _extract_frames(video_path: Path, output_dir: Path, duration: float | None) -> list[Path]:
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    if duration and duration > 1:
        timestamps = [duration * 0.2, duration * 0.5, duration * 0.8]
    else:
        timestamps = [0.0, 0.5, 1.0]

    frames: list[Path] = []
    for i, timestamp in enumerate(timestamps, start=1):
        frame_path = frame_dir / f"frame_{i}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{max(timestamp, 0):.2f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                "scale='min(1280,iw)':-2",
                str(frame_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if frame_path.exists():
            frames.append(frame_path)
    return frames


def _guess_prompt(filename: str, duration: float | None) -> str:
    duration_text = f"{duration:.2f}s" if duration is not None else "unknown"
    return (
        "You are an experienced tennis coach and video analyst. "
        f"You are analyzing a short video clip named {filename} with duration {duration_text}. "
        "First determine whether the sport is tennis. "
        "If it is tennis, analyze it like a coach giving useful hackathon-demo feedback from limited visual evidence. "
        "Do not invent details that are not visible. If uncertain, say 'uncertain' and lower confidence. "
        "Return strict JSON only. "
        "Use exactly these keys: "
        "summary, sport, stroke_type, hitting_phase, player_count, court_context, "
        "key_events, stance, footwork, swing_path, contact_point_estimate, follow_through, "
        "strengths, issues, coaching_tips, overall_assessment, confidence. "
        "Rules: "
        "sport must be a short string. "
        "stroke_type should be values like forehand, backhand, serve, volley, smash, or unknown. "
        "hitting_phase should describe preparation, contact, recovery, or mixed. "
        "player_count should be a short string like '1' or '2'. "
        "court_context should mention baseline, practice wall, match court, unknown, etc. "
        "key_events, strengths, issues, and coaching_tips must be arrays of short strings. "
        "stance, footwork, swing_path, contact_point_estimate, follow_through, and overall_assessment should be short coaching-oriented phrases. "
        "If the clip is not tennis, still return the same JSON schema but keep tennis-specific fields as 'unknown' or empty arrays."
    )


def _analyze_with_openrouter(
    video_path: Path,
    output_dir: Path,
    filename: str,
    duration: float | None,
) -> dict:
    api_key = os.getenv("VIDEO_ANALYZER_API_KEY")
    api_url = os.getenv("VIDEO_ANALYZER_API_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("VIDEO_ANALYZER_MODEL", "meta-llama/llama-3.2-11b-vision-instruct:free")

    if not api_key:
        raise HTTPException(status_code=500, detail="Missing VIDEO_ANALYZER_API_KEY")

    frames = _extract_frames(video_path, output_dir, duration)
    if not frames:
        raise HTTPException(status_code=500, detail="Failed to extract video frames")

    content: list[dict] = [{"type": "text", "text": _guess_prompt(filename, duration)}]
    for frame in frames:
        encoded = base64.b64encode(frame.read_bytes()).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def send_request(request_payload: dict) -> requests.Response:
        last_response = None
        for attempt in range(4):
            response = requests.post(
                f"{api_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=request_payload,
                timeout=180,
            )
            last_response = response
            if response.status_code != 429:
                return response
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
        return last_response

    response = send_request(payload)

    if response.status_code >= 400:
        body_text = response.text
        if "JSON mode is not enabled" in body_text:
            fallback_payload = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
            }
            response = send_request(fallback_payload)

    if response.status_code >= 400:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "OpenRouter request failed",
                "status_code": response.status_code,
                "body": response.text[-4000:],
            },
        )

    data = response.json()
    try:
        message = data["choices"][0]["message"]["content"]
        if isinstance(message, list):
            text_parts = []
            for item in message:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            message = "\n".join(text_parts)
        if not isinstance(message, str):
            raise ValueError("Model response content was not text")
        start = message.find("{")
        end = message.rfind("}")
        if start != -1 and end != -1 and end > start:
            message = message[start : end + 1]
        return json.loads(message)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"message": "Failed to parse model response", "body": data},
        ) from exc


def _download_video(video_url: str, output_path: Path) -> None:
    with requests.get(
        video_url,
        stream=True,
        timeout=180,
        headers={"User-Agent": USER_AGENT},
    ) as response:
        if response.status_code >= 400:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download video: HTTP {response.status_code}",
            )

        size = 0
        with output_path.open("wb") as buffer:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail="Remote video file too large")
                buffer.write(chunk)


def _format_trigger_text(
    analysis: dict[str, Any],
    memory: dict[str, Any],
    context_agent_context: dict[str, Any] | None,
) -> str:
    stroke_type = analysis.get("stroke_type", "unknown")
    overall = analysis.get("overall_assessment", analysis.get("summary", "No assessment available."))
    strengths = analysis.get("strengths") or []
    issues = analysis.get("issues") or []
    tips = analysis.get("coaching_tips") or []
    history = memory.get("history") or []
    stance = analysis.get("stance", "unknown")
    footwork = analysis.get("footwork", "unknown")
    swing_path = analysis.get("swing_path", "unknown")
    contact_point = analysis.get("contact_point_estimate", "unknown")
    follow_through = analysis.get("follow_through", "unknown")
    hitting_phase = analysis.get("hitting_phase", "unknown")
    court_context = analysis.get("court_context", "unknown")

    parts = [
        f"This clip appears to show {stroke_type} tennis work in a {court_context} setting.",
        f"The main phase captured here is {hitting_phase}.",
        f"My overall technical assessment is: {overall}",
    ]

    technical_observations = []
    if stance != "unknown":
        technical_observations.append(f"stance looks {stance}")
    if footwork != "unknown":
        technical_observations.append(f"footwork looks {footwork}")
    if swing_path != "unknown":
        technical_observations.append(f"swing path looks {swing_path}")
    if contact_point != "unknown":
        technical_observations.append(f"contact point appears {contact_point}")
    if follow_through != "unknown":
        technical_observations.append(f"follow through looks {follow_through}")
    if technical_observations:
        parts.append("From a stroke mechanics perspective, " + ", ".join(technical_observations[:4]) + ".")

    if strengths:
        parts.append("The strongest parts of this swing are " + ", ".join(strengths[:3]) + ".")
    if issues:
        parts.append("The main technical issues I would prioritize are " + ", ".join(issues[:3]) + ".")
    if tips:
        parts.append("My coaching recommendation is to " + ", then ".join(tips[:3]) + ".")

    if history:
        previous = history[-1].get("analysis", {})
        previous_stroke = previous.get("stroke_type")
        if previous_stroke and previous_stroke != stroke_type:
            parts.append(f"Compared with your last clip, this one looks more like {stroke_type} than {previous_stroke}.")
        previous_issue = (previous.get("issues") or [None])[0]
        current_issue = issues[0] if issues else None
        if previous_issue and current_issue and previous_issue == current_issue:
            parts.append(f"The same recurring issue still appears to be {current_issue}.")

    if context_agent_context:
        preferred_focus = context_agent_context.get("preferred_focus")
        recurring_issue = context_agent_context.get("recurring_issue")
        dominant_hand = context_agent_context.get("dominant_hand")

        if dominant_hand:
            parts.append(f"I am also using your {dominant_hand}-handed player profile.")
        if recurring_issue:
            parts.append(f"Your recurring pattern has been {recurring_issue}.")
        if preferred_focus:
            parts.append(f"So the main focus for this session should be {preferred_focus}.")

    return " ".join(part.strip() for part in parts if part and str(part).strip())


def _build_updated_context(
    user_id: str,
    analysis: dict[str, Any],
    memory: dict[str, Any],
    context_agent_context: dict[str, Any] | None,
) -> dict[str, Any]:
    previous_context = context_agent_context or {}
    updated_context = {
        **previous_context,
        "user_id": user_id,
        "last_sport": analysis.get("sport"),
        "last_stroke_type": analysis.get("stroke_type"),
        "last_summary": analysis.get("summary"),
        "last_overall_assessment": analysis.get("overall_assessment"),
        "recurring_issue": (analysis.get("issues") or [previous_context.get("recurring_issue")])[0],
        "preferred_focus": (analysis.get("coaching_tips") or [previous_context.get("preferred_focus")])[0],
        "history_count": len(memory.get("history") or []),
    }
    return updated_context


def _maybe_forward_to_external_trigger(
    user_id: str,
    analysis: dict[str, Any],
    memory: dict[str, Any],
    context_agent_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    trigger_url = os.getenv("TRIGGER_URL")
    if not trigger_url:
        return None

    response = requests.post(
        trigger_url,
        json={
            "user_id": user_id,
            "analysis": analysis,
            "memory": memory,
            "context": context_agent_context,
        },
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "External trigger request failed",
                "status_code": response.status_code,
                "body": response.text[-4000:],
            },
        )
    return response.json()


def _run_trigger(user_id: str, analysis: dict[str, Any]) -> dict[str, Any]:
    memory = _load_memory(user_id)
    context_agent_context = _fetch_context_agent_context(user_id)
    external_result = _maybe_forward_to_external_trigger(user_id, analysis, memory, context_agent_context)

    if external_result is not None:
        text = external_result.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(status_code=500, detail="External trigger response missing text")
        result = external_result
    else:
        result = {
            "text": _format_trigger_text(analysis, memory, context_agent_context),
            "metadata": {
                "source": "local-trigger",
                "analysis_summary": analysis.get("summary"),
            },
        }

    history = memory.get("history") or []
    history.append(
        {
            "analysis": analysis,
            "text": result.get("text"),
        }
    )
    memory["history"] = history[-10:]
    _save_memory(user_id, memory)
    updated_context = _build_updated_context(user_id, analysis, memory, context_agent_context)
    _log_to_context_agent(user_id, updated_context, analysis, result.get("text", ""))
    result["metadata"] = {
        **(result.get("metadata") or {}),
        "context_agent_used": context_agent_context is not None,
    }
    return result


def _analyze_local_video(input_path: Path, filename: str) -> dict[str, Any]:
    _ensure_video_suffix(filename)

    with tempfile.TemporaryDirectory(prefix="video-analyzer-") as temp_dir:
        temp_path = Path(temp_dir)
        staged_input_path = temp_path / filename
        output_dir = temp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, staged_input_path)

        duration = _run_ffprobe(staged_input_path)
        if duration is not None and duration > MAX_VIDEO_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Video too long: {duration:.2f}s, max is {MAX_VIDEO_SECONDS}s",
            )

        if shutil.which("video-analyzer") is not None:
            command = _build_analyzer_command(staged_input_path, output_dir)

            try:
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
            except subprocess.TimeoutExpired as exc:
                raise HTTPException(status_code=504, detail="Analysis timed out") from exc
            except subprocess.CalledProcessError as exc:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": "video-analyzer execution failed",
                        "stderr": exc.stderr[-4000:],
                    },
                ) from exc

            analysis_file = output_dir / ANALYZER_OUTPUT_NAME
            if not analysis_file.exists():
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": "analysis.json not found",
                        "stdout": result.stdout[-4000:],
                        "stderr": result.stderr[-4000:],
                    },
                )

            with analysis_file.open() as f:
                analysis = json.load(f)
        else:
            analysis = _analyze_with_openrouter(
                video_path=staged_input_path,
                output_dir=output_dir,
                filename=filename,
                duration=duration,
            )

        return {
            "duration_seconds": duration,
            "filename": filename,
            "analysis": analysis,
        }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/trigger")
def trigger(request: TriggerRequest) -> dict[str, Any]:
    return _run_trigger(request.user_id, request.analysis)


@app.post("/analyze")
def analyze_from_url(request: AnalyzeVideoRequest) -> dict[str, Any]:
    filename = Path(request.video_url).name or "video.mp4"
    if Path(filename).suffix.lower() not in {".mp4", ".mov", ".m4v", ".webm", ".avi"}:
        filename = f"{filename}.mp4"

    with tempfile.TemporaryDirectory(prefix="video-fetch-") as temp_dir:
        input_path = Path(temp_dir) / filename
        _download_video(request.video_url, input_path)
        analysis_result = _analyze_local_video(input_path, filename)

    trigger_result = _run_trigger(request.user_id, analysis_result["analysis"])
    return {
        "text": trigger_result["text"],
        "analysis": analysis_result["analysis"],
        "metadata": {
            "user_id": request.user_id,
            "filename": analysis_result["filename"],
            "duration_seconds": analysis_result["duration_seconds"],
            **(trigger_result.get("metadata") or {}),
        },
    }


@app.post("/analyze-upload")
async def analyze_video(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    with tempfile.TemporaryDirectory(prefix="video-analyzer-") as temp_dir:
        input_path = Path(temp_dir) / file.filename

        size = 0
        with input_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail="File too large")
                buffer.write(chunk)
        return _analyze_local_video(input_path, file.filename)
