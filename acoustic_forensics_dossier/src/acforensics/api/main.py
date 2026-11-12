"""FastAPI Application for Acoustic Forensics Signal Intelligence Lab.

Provides endpoints for:
- POST /api/analyze (multipart audio) -> job_id
- GET  /api/analyze/{id}/events (Server-Sent Events streaming real pipeline stages)
- GET  /api/result/{id} (verdict, probabilities, model breakdown, downsampled spectrograms)
- POST /api/stress-test (re-encode clip with ffmpeg across MP3/AAC/Opus at 16/32/64 kbps)
- GET  /api/benchmarks (dynamically serve measured results from results/ tables/metrics)
- GET  /api/samples & GET /api/samples/{name} (bundled reference clips)
- GET  /api/health (system health, device, loaded models)
- Mounts React production single-page application from frontend/dist if available.

"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import uuid
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from acforensics.api.models import (
    AnalysisResult,
    AnalyzeResponse,
    BenchmarksResponse,
    HealthResponse,
    SampleItem,
    StressTestResponse,
)
from acforensics.api.service import ForensicService, get_forensic_service
from acforensics.utils.logging import get_logger

logger = get_logger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Acoustic Forensics backend service...")
    service = get_forensic_service()
    yield
    logger.info("Shutting down Acoustic Forensics backend service.")


app = FastAPI(
    title="Acoustic Forensics API",
    description="Precision Audio Forensic & Synthetic Speech Detection System",
    version="2.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Returns operational status, compute device, and loaded model information."""
    service = get_forensic_service()
    return service.get_health()


@app.get("/api/samples", response_model=list[SampleItem])
async def list_sample_audio_files():
    """Lists bundled forensic reference audio files available for immediate analysis."""
    service = get_forensic_service()
    return service.list_samples()


@app.get("/api/samples/{filename}")
async def stream_sample_audio(filename: str):
    """Streams a bundled audio reference file for client-side playback."""
    service = get_forensic_service()
    file_path = service.get_sample_file_path(filename)
    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample file '{filename}' was not found.",
        )
    media_type = "audio/wav"
    if filename.endswith(".flac"):
        media_type = "audio/flac"
    elif filename.endswith(".mp3"):
        media_type = "audio/mpeg"
    return FileResponse(file_path, media_type=media_type, filename=filename)


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_audio_upload(
    audio: UploadFile = File(...),
):
    """Uploads an audio file and initializes the forensic analysis job pipeline.

    Validates format and size, assigns a unique Job ID, and prepares the SSE stream.
    """
    filename = audio.filename or "upload.wav"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await audio.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum permissible size of 25MB (Received {len(content) / (1024*1024):.2f}MB).",
        )
    if len(content) < 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File payload is too small to contain valid audio data.",
        )

    job_id = uuid.uuid4().hex
    service = get_forensic_service()
    await service.register_audio_upload(job_id, content, filename)

    return AnalyzeResponse(
        job_id=job_id,
        status="processing",
        filename=filename,
        stream_url=f"/api/analyze/{job_id}/events",
        result_url=f"/api/result/{job_id}",
    )


@app.post("/api/analyze/sample/{sample_id}", response_model=AnalyzeResponse)
async def analyze_sample_by_id(sample_id: str):
    """Convenience endpoint to initiate analysis directly on a bundled reference sample."""
    service = get_forensic_service()
    samples = service.list_samples()
    target_sample = next((s for s in samples if s.id == sample_id or s.filename == sample_id), None)
    if not target_sample:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample '{sample_id}' not found. Available samples: {[s.id for s in samples]}",
        )

    file_path = service.get_sample_file_path(target_sample.filename)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample audio file missing from disk.")

    with open(file_path, "rb") as f:
        content = f.read()

    job_id = uuid.uuid4().hex
    await service.register_audio_upload(job_id, content, target_sample.filename)

    return AnalyzeResponse(
        job_id=job_id,
        status="processing",
        filename=target_sample.filename,
        stream_url=f"/api/analyze/{job_id}/events",
        result_url=f"/api/result/{job_id}",
    )


@app.get("/api/analyze/{job_id}/events")
async def stream_analysis_events(job_id: str):
    """Server-Sent Events (SSE) stream emitting real pipeline stages in sequence:

    1. validating -> 2. resampling -> 3. feature_extraction -> 4. model_scores -> 5. ensemble -> 6. done
    """
    service = get_forensic_service()

    async def event_generator():
        async for event in service.stream_pipeline_analysis(job_id):
            yield {
                "event": "pipeline_event",
                "data": event.model_dump_json(),
            }

    return EventSourceResponse(event_generator())


@app.get("/api/result/{job_id}", response_model=AnalysisResult)
async def get_analysis_result(job_id: str):
    """Fetches the full forensic result report, downsampled spectrograms, and attention map."""
    service = get_forensic_service()
    res = service.get_result(job_id)
    if res is None:
        if job_id in service.cached_audio:
            async for _ in service.stream_pipeline_analysis(job_id):
                pass
            res = service.get_result(job_id)

    if res is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis result for job '{job_id}' not found or has not finished processing.",
        )
    return res


@app.post("/api/stress-test", response_model=StressTestResponse)
async def run_codec_stress_test(job_id: str = Form(...)):
    """Re-encodes the specified audio across MP3, AAC, and Opus at 16, 32, 64 kbps via ffmpeg

    and computes the comparative verdict shifts to assess acoustic robustness.
    """
    service = get_forensic_service()
    try:
        return await service.run_stress_test(job_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Codec stress test failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Stress test error: {str(e)}")


@app.get("/api/benchmarks", response_model=BenchmarksResponse)
async def get_experimental_benchmarks():
    """Serves official ASVspoof 2019 LA evaluation tables, ablations, and ROC curves

    dynamically parsed from results/ without hardcoded parameters.
    """
    service = get_forensic_service()
    try:
        return service.load_benchmarks()
    except Exception as e:
        logger.exception("Failed to load benchmark results: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


repo_root = Path(__file__).resolve().parents[3]
frontend_dist = repo_root / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    logger.info("Mounting frontend SPA from: %s", frontend_dist)
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
