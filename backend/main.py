import os, uuid, json, subprocess, re, traceback, shutil, threading
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
BASE_DIR = Path(__file__).parent.parent
CLIPS_DIR = BASE_DIR / "clips"
UPLOADS_DIR = BASE_DIR / "uploads"
CLIPS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend")), name="static")

_groq_client = None

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client

JOB_DIR = Path("/tmp/jobs")
JOB_DIR.mkdir(parents=True, exist_ok=True)


def _job_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def read_job(job_id: str) -> dict | None:
    try:
        return json.loads(_job_path(job_id).read_text())
    except FileNotFoundError:
        return None


def write_job(job_id: str, data: dict):
    _job_path(job_id).write_text(json.dumps(data))


def update_job(job_id: str, **kwargs):
    data = read_job(job_id) or {}
    data.update(kwargs)
    write_job(job_id, data)


class VideoRequest(BaseModel):
    url: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(str(BASE_DIR / "frontend" / "index.html"))


@app.post("/process")
async def process(req: VideoRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    write_job(job_id, {
        "status": "queued",
        "progress": 0,
        "message": "Job queued...",
        "clips": [],
        "error": None,
    })
    bg.add_task(run_pipeline, job_id, req.url)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = read_job(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return job


@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    _job_path(job_id).unlink(missing_ok=True)
    shutil.rmtree(UPLOADS_DIR / job_id, ignore_errors=True)
    return {"ok": True}


def _delete_later(path: Path, delay: int = 3600):
    """Delete a directory tree after `delay` seconds (daemon thread — won't block shutdown)."""
    def _run():
        threading.Event().wait(delay)
        shutil.rmtree(path, ignore_errors=True)
    t = threading.Thread(target=_run, daemon=True)
    t.start()


def upd(job_id: str, status: str, progress: int, message: str):
    update_job(job_id, status=status, progress=progress, message=message)


def run_pipeline(job_id: str, url: str):
    job_dir = UPLOADS_DIR / job_id
    clip_dir = CLIPS_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    clip_dir.mkdir(exist_ok=True)

    try:
        upd(job_id, "downloading", 5, "Downloading video...")
        video_path = download_video(url, job_dir)

        upd(job_id, "extracting", 20, "Extracting audio track...")
        audio_path = job_dir / "audio.wav"
        run_cmd([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ar", "16000", "-ac", "1", "-f", "wav", str(audio_path),
        ])

        upd(job_id, "transcribing", 35, "Transcribing speech (this may take a while)...")
        segments = transcribe_audio(audio_path)
        if not segments:
            raise ValueError("No speech detected in the video.")
        transcript = build_transcript(segments)

        upd(job_id, "analyzing", 60, "AI analyzing for viral moments...")
        moments = pick_viral_moments(transcript)
        if not moments:
            update_job(job_id,
                status="done", progress=100,
                message="No viral moments found in this video.", clips=[],
            )
            return

        clips = []
        for i, moment in enumerate(moments):
            upd(job_id, "cutting", 65 + i * 6, f"Cutting clip {i + 1} of {len(moments)}...")
            out_path = clip_dir / f"clip_{i + 1}.mp4"
            start = float(moment["start"])
            duration = 60.0

            cut_clip(
                video_path=video_path,
                start=start,
                duration=duration,
                out_path=out_path,
            )
            clips.append({
                "title": moment.get("title", f"Clip {i + 1}"),
                "hook": moment.get("hook", ""),
                "url": f"/clips/{job_id}/clip_{i + 1}.mp4",
                "duration": 60,
                "start": round(start),
            })

        # Immediately remove the raw download/audio — clips are all that's needed now
        shutil.rmtree(job_dir, ignore_errors=True)
        # Schedule clip folder removal 1 hour after the job completes
        _delete_later(clip_dir, delay=3600)

        update_job(job_id, status="done", progress=100, message="All clips ready!", clips=clips)

    except Exception as exc:
        traceback.print_exc()
        shutil.rmtree(job_dir, ignore_errors=True)
        msg = str(exc).strip() or type(exc).__name__
        if "CalledProcessError" in msg:
            msg = "ffmpeg error — video may be unsupported or too short."
        update_job(job_id, status="error", progress=0, message=msg, error=msg)


def _resolve_cookies() -> str | None:
    """Return a path to a cookies file, or None if unavailable.
    Prefers the YOUTUBE_COOKIES env var (Railway) over a local file."""
    content = os.getenv("YOUTUBE_COOKIES", "").strip()
    if content:
        tmp = Path("/tmp/yt_cookies.txt")
        tmp.write_text(content)
        return str(tmp)
    local = BASE_DIR / "cookies.txt"
    if local.exists():
        return str(local)
    return None


def download_video(url: str, out_dir: Path) -> Path:
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--no-check-certificates",
        "--extractor-args", "youtube:player_client=ios;formats=missing_pot",
        "-o", str(out_dir / "video.%(ext)s"),
    ]
    cookies_file = _resolve_cookies()
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode().strip()
        stdout = result.stdout.decode().strip()
        detail = stderr or stdout or "yt-dlp exited with a non-zero code"
        raise ValueError(f"Download failed: {detail[-400:]}")

    for f in out_dir.iterdir():
        if f.stem == "video":
            return f
    raise ValueError("Downloaded file not found after yt-dlp.")


def run_cmd(cmd: list):
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode().strip()
        stdout = result.stdout.decode().strip()
        detail = stderr or stdout or f"{cmd[0]} exited with code {result.returncode}"
        raise RuntimeError(detail[-400:])


def transcribe_audio(audio_path: Path) -> list:
    with open(audio_path, "rb") as f:
        transcription = get_groq_client().audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    return [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in (transcription.segments or [])
        if s.text.strip()
    ]


def build_transcript(segments: list) -> str:
    lines = [f"[{s['start']:.1f}s-{s['end']:.1f}s] {s['text']}" for s in segments]
    return "\n".join(lines)


def pick_viral_moments(transcript: str) -> list:
    system_prompt = (
        "You are a brutally selective viral content editor. Your job is to find moments "
        "that will genuinely stop someone mid-scroll. Return ONLY valid JSON — no markdown, no explanation."
    )
    user_prompt = (
        "Analyze this transcript and identify ONLY the genuinely viral moments — moments that are "
        "shocking, unexpectedly funny, deeply emotional, controversial, or have an irresistible hook.\n\n"
        "RULES:\n"
        "- Be ruthlessly selective. A boring video might only have 1 viral moment, or zero.\n"
        "- Never invent or pad clips just to reach a number. Quality over quantity, always.\n"
        "- Each clip will be exactly 60 seconds. Pick a start time that centers the peak moment.\n"
        "- Clips must not overlap. Leave at least 30 seconds between clip end times and the next start.\n"
        "- If there are no genuinely viral moments, return an empty array [].\n\n"
        "Return a JSON array (length 0 to 8) where each object has:\n"
        '- "title": punchy clip title (max 8 words, no fluff)\n'
        '- "hook": one sentence explaining exactly why this moment goes viral\n'
        '- "start": start time in seconds (number) — position this so the best 60s are captured\n\n'
        f"Transcript:\n{transcript[:8000]}"
    )

    resp = get_groq_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    raw = resp.choices[0].message.content.strip()
    # strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    # find the JSON array in the response
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    moments = json.loads(raw)
    if not isinstance(moments, list):
        raise ValueError("Groq returned unexpected format.")
    return moments


def cut_clip(video_path: Path, start: float, duration: float, out_path: Path):
    """Dispatcher: smart subject-tracking crop, falls back to letterbox."""
    try:
        import cv2
        import numpy as np
        _cut_clip_smart(video_path, start, duration, out_path, cv2, np)
    except Exception:
        traceback.print_exc()
        _cut_clip_letterbox(video_path, start, duration, out_path)


def _cut_clip_letterbox(video_path: Path, start: float, duration: float, out_path: Path):
    vf = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    run_cmd([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ])


def _cut_clip_smart(video_path: Path, start: float, duration: float, out_path: Path, cv2, np):
    TARGET_W, TARGET_H = 1080, 1920
    ASPECT = TARGET_W / TARGET_H  # 9/16 = 0.5625

    cap = cv2.VideoCapture(str(video_path))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Largest 9:16 crop box that fits inside the source frame
    if vid_w / vid_h > ASPECT:
        crop_h = vid_h
        crop_w = int(crop_h * ASPECT) & ~1
    else:
        crop_w = vid_w & ~1
        crop_h = int(crop_w / ASPECT) & ~1

    total_frames = max(1, int(duration * fps))
    SAMPLE_EVERY = max(1, int(fps / 2))  # analyse 2 frames/sec

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    sampled: dict = {}
    for fi in range(total_frames):
        if fi % SAMPLE_EVERY == 0:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                cx = fx + fw // 2
                cy = fy + int(fh * 0.3)  # slightly above face center for natural headroom
            else:
                # Brightness-weighted centroid on downsampled frame
                small = cv2.resize(gray, (320, max(1, int(320 * vid_h / vid_w))))
                blurred = cv2.GaussianBlur(small.astype(np.float32), (11, 11), 0)
                total_w = float(blurred.sum()) or 1.0
                sh, sw = small.shape
                ys, xs = np.mgrid[0:sh, 0:sw]
                cx = int((xs * blurred).sum() / total_w * vid_w / sw)
                cy = int((ys * blurred).sum() / total_w * vid_h / sh)
            sampled[fi] = (cx, cy)
        else:
            if not cap.grab():
                break

    cap.release()

    if not sampled:
        sampled[0] = (vid_w // 2, vid_h // 2)

    # Interpolate to every frame
    idx = sorted(sampled)
    raw_cx = np.interp(range(total_frames), idx, [sampled[i][0] for i in idx])
    raw_cy = np.interp(range(total_frames), idx, [sampled[i][1] for i in idx])

    # EMA smoothing — ~1-second lag prevents jitter while still following subject
    ALPHA = 0.04
    cx_s = np.empty(total_frames)
    cy_s = np.empty(total_frames)
    cx_s[0], cy_s[0] = raw_cx[0], raw_cy[0]
    for i in range(1, total_frames):
        cx_s[i] = ALPHA * raw_cx[i] + (1 - ALPHA) * cx_s[i - 1]
        cy_s[i] = ALPHA * raw_cy[i] + (1 - ALPHA) * cy_s[i - 1]

    # Convert center to top-left crop origin, clamped to video bounds
    x_t = np.clip(cx_s - crop_w // 2, 0, vid_w - crop_w).astype(int)
    y_t = np.clip(cy_s - crop_h // 2, 0, vid_h - crop_h).astype(int)

    def build_expr(track: np.ndarray) -> str:
        """Nested if(lt(t,T),V,…) expression with ≤60 keypoints."""
        n = len(track)
        step = max(1, n // 60)
        pts = [(round(i / fps, 3), int(track[min(i, n - 1)])) for i in range(0, n, step)]
        expr = str(pts[-1][1])
        for t, v in reversed(pts[:-1]):
            expr = f"if(lt(t,{t}),{v},{expr})"
        return expr

    vf = (
        f"crop={crop_w}:{crop_h}:{build_expr(x_t)}:{build_expr(y_t)},"
        f"scale={TARGET_W}:{TARGET_H}:flags=lanczos,"
        f"setsar=1"
    )
    run_cmd([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ])
