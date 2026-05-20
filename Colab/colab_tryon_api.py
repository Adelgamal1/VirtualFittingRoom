import argparse
import base64
import io
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw


APP = FastAPI(title="Virtual Fitting Room Colab API")
RUNTIME = None
RUNTIME_LOCK = threading.Lock()

APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def find_project_bundle(project_root: Path) -> Path:
    if (project_root / "model" / "pipeline.py").exists():
        return project_root
    if (project_root / "pipeline.py").exists() and project_root.name == "model":
        return project_root.parent

    for candidate in project_root.iterdir():
        if candidate.is_dir() and (candidate / "model" / "pipeline.py").exists():
            return candidate

    raise FileNotFoundError(f"Could not find CatVTON model/pipeline.py under {project_root}.")


def build_fallback_mask(image: Image.Image, category: str) -> Image.Image:
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if category == "lower":
        box = (int(width * 0.20), int(height * 0.40), int(width * 0.80), int(height * 0.98))
    elif category == "overall":
        box = (int(width * 0.14), int(height * 0.05), int(width * 0.86), int(height * 0.98))
    else:
        box = (int(width * 0.14), int(height * 0.07), int(width * 0.86), int(height * 0.62))

    draw.rounded_rectangle(box, radius=max(16, width // 14), fill=255)
    return mask


def normalize_category(category: str) -> str:
    value = (category or "upper").strip().lower()
    if value in {"t-shirt", "tshirt", "tee", "shirt", "hoodie", "jacket", "upper_body"}:
        return "upper"
    if value in {"pants", "trousers", "jeans", "shorts", "lower_body"}:
        return "lower"
    if value in {"dress", "dresses", "galabeya", "galabiya", "overall"}:
        return "overall"
    return value if value in {"upper", "lower", "overall"} else "upper"


def load_runtime(args):
    project_root = Path(args.project_root).expanduser().resolve()
    bundle_root = find_project_bundle(project_root)

    if str(bundle_root) not in sys.path:
        sys.path.insert(0, str(bundle_root))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from model.pipeline import CatVTONPipeline

    try:
        from model.cloth_masker import AutoMasker
    except Exception as ex:
        AutoMasker = None
        print(f"AutoMasker import failed; fallback mask will be used: {ex}", flush=True)

    resume_path = args.resume_path
    if not resume_path:
        resume_path = snapshot_download("zhengchong/CatVTON")

    densepose_root = Path(resume_path) / "DensePose"
    schp_root = Path(resume_path) / "SCHP"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    masker = None
    if AutoMasker and densepose_root.exists() and schp_root.exists():
        masker = AutoMasker(
            densepose_ckpt=str(densepose_root),
            schp_ckpt=str(schp_root),
            device=device,
        )

    pipeline = CatVTONPipeline(
        base_ckpt=args.base_model_path,
        attn_ckpt=resume_path,
        attn_ckpt_version=args.attn_version,
        weight_dtype=dtype,
        device=device,
        skip_safety_check=True,
    )

    return {
        "pipeline": pipeline,
        "masker": masker,
        "device": device,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
    }


def get_runtime(args):
    global RUNTIME
    with RUNTIME_LOCK:
        if RUNTIME is None:
            RUNTIME = load_runtime(args)
        return RUNTIME


def run_tryon(args, person_bytes: bytes, garment_bytes: bytes, category: str) -> bytes:
    runtime = get_runtime(args)
    category = normalize_category(category)

    person = Image.open(io.BytesIO(person_bytes)).convert("RGB")
    garment = Image.open(io.BytesIO(garment_bytes)).convert("RGB")

    if runtime["masker"] is not None:
        mask = runtime["masker"](person, mask_type=category)["mask"]
    else:
        mask = build_fallback_mask(person, category)

    generator = torch.Generator(device=runtime["device"]).manual_seed(args.seed)
    with torch.inference_mode():
        result = runtime["pipeline"](
            person,
            garment,
            mask,
            num_inference_steps=runtime["steps"],
            guidance_scale=runtime["guidance_scale"],
            height=runtime["height"],
            width=runtime["width"],
            generator=generator,
        )[0]

    output = io.BytesIO()
    result.save(output, format="PNG")
    return output.getvalue()


def install_cloudflared(path: Path):
    if path.exists():
        return
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    subprocess.check_call(["wget", "-q", "-O", str(path), url])
    subprocess.check_call(["chmod", "+x", str(path)])


def start_tunnel(port: int):
    cloudflared = Path("/content/cloudflared")
    install_cloudflared(cloudflared)
    process = subprocess.Popen(
        [str(cloudflared), "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    public_url = None

    def reader():
        nonlocal public_url
        for line in process.stdout:
            print(line, end="", flush=True)
            match = re.search(r"https://[-a-zA-Z0-9.]+\.trycloudflare\.com", line)
            if match:
                public_url = match.group(0)
                print("\nCOPY THIS INTO appsettings.json:", flush=True)
                print(f"VirtualTryOn:ApiUrl = {public_url}/tryon\n", flush=True)

    threading.Thread(target=reader, daemon=True).start()

    deadline = time.time() + 60
    while public_url is None and time.time() < deadline:
        time.sleep(1)

    return public_url


def create_app(args):
    @APP.on_event("startup")
    def startup_load_model():
        print("Loading virtual try-on model. Wait for 'Model is ready' before using the website.", flush=True)
        get_runtime(args)
        print("Model is ready. You can now use the /tryon URL from the website.", flush=True)

    @APP.get("/health")
    def health():
        return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}

    @APP.post("/tryon")
    async def tryon(
        person_image: UploadFile = File(...),
        garment_image: UploadFile = File(...),
        category: str = Form("upper"),
    ):
        try:
            person_bytes = await person_image.read()
            garment_bytes = await garment_image.read()
            output = run_tryon(args, person_bytes, garment_bytes, category)
            return Response(content=output, media_type="image/png")
        except Exception as ex:
            return JSONResponse(status_code=500, content={"error": str(ex)})

    return APP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/content/CatVTON")
    parser.add_argument("--base-model-path", default="booksforcharlie/stable-diffusion-inpainting")
    parser.add_argument("--resume-path", default="")
    parser.add_argument("--attn-version", default="mix")
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tunnel", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    create_app(ARGS)
    if ARGS.tunnel:
        start_tunnel(ARGS.port)
    uvicorn.run(APP, host=ARGS.host, port=ARGS.port)
