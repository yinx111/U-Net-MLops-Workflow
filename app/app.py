#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import tifffile as tiff
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
TEMPLATE_PATH = ROOT_DIR / "index.html"
RUNS_DIR = PROJECT_ROOT / "webapp_runs"
INFER_SCRIPT = PROJECT_ROOT / "scripts" / "inference.py"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "model.pth"

HOST = os.getenv("APP_HOST", "0.0.0.0")
PORT = int(os.getenv("APP_PORT", "8000"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT_SECONDS", "3600"))
ALLOWED_SUFFIX = {".tif", ".tiff", ".png"}

app = FastAPI(title="U-Net++ Inference App", version="1.0.0")
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _render_index() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def _zip_dir(source_dir: Path, zip_path: Path) -> int:
    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(source_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(source_dir))
                count += 1
    return count


def _to_preview_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    # Convert CHW -> HWC when needed
    if arr.shape[0] in (1, 3, 4, 5) and arr.shape[0] < arr.shape[1]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError(f"Unsupported channel shape: {arr.shape}")

    arr = arr.astype(np.float32)
    v_min = float(np.nanmin(arr))
    v_max = float(np.nanmax(arr))
    if v_max <= v_min:
        arr = np.zeros_like(arr, dtype=np.float32)
    else:
        arr = (arr - v_min) / (v_max - v_min + 1e-6)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def _run_inference(input_path: Path, out_dir: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["FORCE_CPU"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["IMG_PATH"] = str(input_path)
    env["OUT_DIR"] = str(out_dir)
    if "CHECKPOINT_PATH" not in env and DEFAULT_MODEL_PATH.exists():
        env["CHECKPOINT_PATH"] = str(DEFAULT_MODEL_PATH)

    cmd = [sys.executable, str(INFER_SCRIPT)]
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=INFERENCE_TIMEOUT,
    )


def _resolve_checkpoint_path() -> Path:
    checkpoint_env = os.getenv("CHECKPOINT_PATH")
    if checkpoint_env:
        return Path(checkpoint_env)
    return DEFAULT_MODEL_PATH


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_render_index())


@app.get("/healthz")
def healthz() -> JSONResponse:
    checkpoint_path = _resolve_checkpoint_path()
    if not checkpoint_path.exists():
        return JSONResponse(
            {
                "status": "error",
                "reason": "checkpoint_missing",
                "checkpoint_path": str(checkpoint_path),
            },
            status_code=503,
        )

    try:
        if checkpoint_path.stat().st_size <= 0:
            raise RuntimeError("checkpoint is empty")
        with checkpoint_path.open("rb") as f:
            f.read(16)
    except Exception as e:
        return JSONResponse(
            {
                "status": "error",
                "reason": "checkpoint_unreadable",
                "checkpoint_path": str(checkpoint_path),
                "detail": str(e),
            },
            status_code=503,
        )

    return JSONResponse({"status": "ok", "checkpoint_path": str(checkpoint_path)})


@app.post("/preview")
def preview(image: UploadFile = File(...)):
    if not image.filename:
        return JSONResponse({"error": "Please upload an image file."}, status_code=400)

    suffix = Path(image.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIX:
        return JSONResponse({"error": "Only .tif/.tiff/.png files are supported."}, status_code=400)

    payload = image.file.read()
    if not payload:
        return JSONResponse({"error": "Uploaded content is empty."}, status_code=400)
    if len(payload) > MAX_UPLOAD_MB * 1024 * 1024:
        return JSONResponse(
            {"error": f"File is too large. Limit is {MAX_UPLOAD_MB}MB."},
            status_code=413,
        )

    try:
        if suffix in {".tif", ".tiff"}:
            arr = tiff.imread(BytesIO(payload))
            rgb = _to_preview_rgb(arr)
            buf = BytesIO()
            Image.fromarray(rgb).save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")

        return Response(content=payload, media_type="image/png")
    except Exception as e:
        return JSONResponse({"error": f"Failed to generate preview: {e}"}, status_code=500)


@app.post("/infer")
def infer(image: UploadFile = File(...)):
    if not image.filename:
        return JSONResponse({"error": "Please upload an image file."}, status_code=400)

    suffix = Path(image.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIX:
        return JSONResponse({"error": "Only .tif/.tiff/.png files are supported."}, status_code=400)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    in_dir = run_dir / "input"
    out_dir = run_dir / "infer"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = in_dir / f"uploaded{suffix}"
    with input_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)
    if input_path.stat().st_size <= 0:
        return JSONResponse({"error": "Uploaded content is empty."}, status_code=400)
    if input_path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        return JSONResponse(
            {"error": f"File is too large. Limit is {MAX_UPLOAD_MB}MB."},
            status_code=413,
        )

    try:
        result = _run_inference(input_path, out_dir)
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "Inference timeout."}, status_code=504)
    except Exception as e:
        return JSONResponse({"error": f"Inference process error: {e}"}, status_code=500)

    if result.returncode != 0:
        err_text = result.stderr or result.stdout or "No error output."
        return JSONResponse({"error": err_text[-5000:]}, status_code=500)

    zip_path = run_dir / "result.zip"
    file_count = _zip_dir(out_dir, zip_path)
    if file_count == 0:
        return JSONResponse(
            {"error": "Inference completed, but no downloadable files were generated."},
            status_code=500,
        )

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"infer_result_{run_id}.zip",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, reload=False)
