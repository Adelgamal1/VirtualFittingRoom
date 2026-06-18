import io
import json
import os
import sys
import threading
from pathlib import Path

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat

# ── Fast downloads ───────────────────────────────────────────────────────────
# Enable the Rust-based hf_transfer backend so snapshot_download pulls model
# files in parallel at much higher throughput. If the `hf_transfer` package
# isn't installed, huggingface_hub just falls back to the normal downloader.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# ── Config ────────────────────────────────────────────────────────────────────

SPACE_ROOT    = Path(__file__).resolve().parent
PROJECT_ROOT  = Path(os.getenv("CATVTON_PROJECT_ROOT", SPACE_ROOT))
BASE_MODEL_PATH_RAW = os.getenv("CATVTON_BASE_MODEL_PATH", "").strip()
RESUME_PATH_RAW     = os.getenv("CATVTON_RESUME_PATH", "").strip()
ATTN_VERSION  = os.getenv("CATVTON_ATTN_VERSION", "mix")
MODEL_REPO_ID = os.getenv("CATVTON_MODEL_REPO_ID", "Adelgamal1/virtual-fitting-room-model")

# Warm-up may need to build the whole pipeline (slow, first time only);
# normal inference duration is now computed dynamically (see _infer_duration)
# based on the requested number of denoise steps.
WARMUP_GPU_DURATION = int(os.getenv("CATVTON_WARMUP_DURATION", "300"))

# ── Path helpers ──────────────────────────────────────────────────────────────

def find_bundle(root: Path) -> Path:
    if (root / "model" / "pipeline.py").exists():
        return root
    for d in root.iterdir():
        if d.is_dir() and (d / "model" / "pipeline.py").exists():
            return d
    raise FileNotFoundError(f"No model bundle found under {root}")


def find_dir(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_dir():
        return direct
    matches = sorted(root.rglob(name), key=str)
    if not matches:
        raise FileNotFoundError(f"Directory '{name}' not found under {root}")
    return matches[-1]

# ── Pose ──────────────────────────────────────────────────────────────────────

def read_pose(data: str, size):
    if not data:
        return None
    try:
        landmarks = json.loads(data).get("landmarks") or []
        W, H = size

        def pt(i, min_vis=0.24):
            if i >= len(landmarks):
                return None
            lm = landmarks[i] or {}
            vis = float(lm.get("visibility", 1.0))
            if vis < min_vis:
                return None
            x, y = float(lm.get("x", 0)), float(lm.get("y", 0))
            if x <= 1.5 and y <= 1.5:
                x, y = x * W, y * H
            return (x, y, vis)

        ls, rs = pt(11, 0.34), pt(12, 0.34)
        lh, rh = pt(23, 0.22), pt(24, 0.22)
        if not all([ls, rs, lh, rh]):
            return None

        ls, rs = sorted([ls, rs], key=lambda p: p[0])
        lh, rh = sorted([lh, rh], key=lambda p: p[0])
        sw = abs(rs[0] - ls[0])
        if not (W * 0.08 < sw < W * 0.78):
            return None

        sc = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hc = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        th = hc[1] - sc[1]
        if th < H * 0.15:
            return None

        return dict(
            left_shoulder=ls, right_shoulder=rs,
            left_hip=lh, right_hip=rh,
            shoulder_center=sc, hip_center=hc,
            shoulder_width=sw, torso_height=th,
            collar_y=sc[1] - sw * 0.09,
            neck_y=sc[1] - sw * 0.18,
        )
    except Exception:
        return None

# ── Masking ───────────────────────────────────────────────────────────────────

def build_mask(image: Image.Image, category: str, pose_data=None) -> Image.Image:
    W, H = image.size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    pose = read_pose(pose_data, image.size)

    if category == "upper":
        if pose:
            sw = pose["shoulder_width"]
            sy  = int(pose["shoulder_center"][1] - sw * 0.075)
            cy  = int(pose["collar_y"])
            hem = int(min(H * 0.84, pose["shoulder_center"][1] + pose["torso_height"] * 0.58 + sw * 0.08))
            cx  = int(pose["shoulder_center"][0] * 0.64 + pose["hip_center"][0] * 0.36)
            lsx, rsx = int(pose["left_shoulder"][0]), int(pose["right_shoulder"][0])
            l   = int(max(0, lsx - sw * 0.46))
            r   = int(min(W, rsx + sw * 0.46))
            bh  = int(max(sw * 0.50, abs(pose["right_hip"][0] - pose["left_hip"][0]) * 0.62))
            draw.polygon([
                (l, int(sy + sw * 0.16)),
                (int(lsx - sw * 0.06), cy),
                (int(cx - sw * 0.22), int(cy - sw * 0.035)),
                (int(cx + sw * 0.22), int(cy - sw * 0.035)),
                (int(rsx + sw * 0.06), cy),
                (r, int(sy + sw * 0.16)),
                (int(cx + bh), hem),
                (int(cx - bh), hem),
            ], fill=255)
            draw.ellipse((int(cx - sw * 0.155), int(cy - sw * 0.075),
                          int(cx + sw * 0.155), int(cy + sw * 0.070)), fill=0)
        else:
            sy, hem, cx = int(H * 0.34), int(H * 0.78), W // 2
            draw.polygon([
                (cx, sy), (int(W * 0.82), int(H * 0.38)),
                (int(W * 0.80), hem), (int(W * 0.20), hem),
                (int(W * 0.18), int(H * 0.38)),
            ], fill=255)
            draw.ellipse((int(W * 0.40), int(H * 0.28), int(W * 0.60), int(H * 0.44)), fill=0)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1.2))

    elif category == "lower":
        draw.rounded_rectangle((int(W*0.22), int(H*0.42), int(W*0.78), int(H*0.96)),
                                radius=max(12, W//16), fill=255)
    else:
        draw.rounded_rectangle((int(W*0.16), int(H*0.06), int(W*0.84), int(H*0.96)),
                                radius=max(12, W//16), fill=255)
    return mask


def protect_identity(mask: Image.Image, category: str, pose_data=None) -> Image.Image:
    if category != "upper":
        return mask
    mask = mask.convert("L")
    W, H = mask.size
    guard = Image.new("L", (W, H), 0)
    draw  = ImageDraw.Draw(guard)
    pose  = read_pose(pose_data, mask.size)

    if pose:
        sw, cy = pose["shoulder_width"], int(pose["collar_y"])
        cx = int(pose["shoulder_center"][0])
        draw.rectangle((0, 0, W, max(0, int(cy - sw * 0.18))), fill=255)
        draw.ellipse((int(cx - sw*0.16), int(cy - sw*0.16),
                      int(cx + sw*0.16), int(cy + sw*0.07)), fill=255)
    else:
        draw.rectangle((0, 0, W, int(H * 0.32)), fill=255)
        draw.ellipse((int(W*0.34), int(H*0.12), int(W*0.66), int(H*0.46)), fill=255)

    mask.paste(0, mask=guard)
    return mask

# ── Result validation ─────────────────────────────────────────────────────────

def _mean_diff(a: Image.Image, b: Image.Image, box) -> float:
    try:
        ca = a.crop(box).resize((64, 64)).convert("RGB")
        cb = b.crop(box).resize((64, 64)).convert("RGB")
        stat = ImageStat.Stat(ImageChops.difference(ca, cb))
        return sum(stat.mean) / 3.0
    except Exception:
        return 0.0


def is_bad_result(result: Image.Image, person: Image.Image, pose_data=None) -> bool:
    result = result.convert("RGB")
    person = person.convert("RGB")
    if result.size != person.size:
        result = result.resize(person.size, Image.LANCZOS)
    W, H = person.size
    pose = read_pose(pose_data, person.size)
    if pose:
        sw, cy = pose["shoulder_width"], int(pose["collar_y"])
        pb = max(1, int(cy - sw * 0.12))
    else:
        pb = max(1, int(H * 0.22))
    return _mean_diff(result, person, (0, 0, W, pb)) > 34 or \
           _mean_diff(result, person, (0, 0, W, int(H * 0.72))) > 82

# ── Fallback renderer ─────────────────────────────────────────────────────────

def garment_bbox(image: Image.Image):
    img = image.convert("RGB")
    W, H = img.size
    x0, y0, x1, y1 = W, H, -1, -1
    px = img.load()
    for y in range(0, H, 2):
        for x in range(0, W, 2):
            r, g, b = px[x, y]
            if r > 238 and g > 238 and b > 238:
                continue
            if max(r,g,b) - min(r,g,b) < 4 and max(r,g,b) > 230:
                continue
            x0, y0 = min(x0,x), min(y0,y)
            x1, y1 = max(x1,x), max(y1,y)
    return (0, 0, W, H) if x1 < x0 else (max(0,x0-4), max(0,y0-4), min(W,x1+5), min(H,y1+5))


def base_color(garment: Image.Image):
    px = list(garment.convert("RGB").resize((180,180)).getdata())
    colors = [c for c in px if not (c[0]>238 and c[1]>238 and c[2]>238)]
    dark   = [c for c in colors if sum(c)/3 < 95]
    sample = sorted(dark if len(dark) > 30 else colors, key=sum)
    if not sample:
        return (24, 24, 27)
    mid = sample[len(sample)//2]
    return tuple(max(0, min(255, int(v))) for v in mid)


def extract_print(garment: Image.Image, bc):
    img  = garment.convert("RGB")
    crop = img.crop(garment_bbox(img))
    W, H = crop.size
    px   = crop.load()
    br, bg, bb = bc
    x0, y0, x1, y1 = W, H, -1, -1
    for y in range(H):
        for x in range(W):
            r, g, b = px[x, y]
            if r > 242 and g > 242 and b > 242:
                continue
            if abs(r-br)+abs(g-bg)+abs(b-bb) < 70:
                continue
            x0, y0 = min(x0,x), min(y0,y)
            x1, y1 = max(x1,x), max(y1,y)

    if x1 < x0:
        x0, x1 = int(W*0.24), int(W*0.76)
        y0, y1 = int(H*0.22), int(H*0.72)
    else:
        px2, py2 = int((x1-x0+1)*0.08), int((y1-y0+1)*0.08)
        x0, x1 = max(0,x0-px2), min(W-1,x1+px2)
        y0, y1 = max(0,y0-py2), min(H-1,y1+py2)

    out = crop.crop((x0, y0, x1+1, y1+1)).convert("RGBA")
    alpha = Image.new("L", out.size, 0)
    apx   = alpha.load()
    opx   = out.load()
    for y in range(out.height):
        for x in range(out.width):
            r, g, b, _ = opx[x, y]
            if not (r > 245 and g > 245 and b > 245):
                apx[x, y] = 255
    out.putalpha(alpha.filter(ImageFilter.GaussianBlur(radius=0.4)))
    return out


def render_fallback(person: Image.Image, garment: Image.Image, pose_data=None) -> Image.Image:
    base = person.convert("RGBA")
    W, H = base.size
    pose = read_pose(pose_data, base.size)
    if not pose:
        return person.convert("RGB")

    sw  = pose["shoulder_width"]
    cx  = int(pose["shoulder_center"][0] * 0.66 + pose["hip_center"][0] * 0.34)
    cy  = int(pose["collar_y"])
    sy  = int(pose["shoulder_center"][1] - sw * 0.075)
    bc_ = base_color(garment)

    shirt_mask = protect_identity(build_mask(person, "upper", pose_data), "upper", pose_data)
    shirt_mask = shirt_mask.filter(ImageFilter.GaussianBlur(radius=0.9))

    layer = Image.new("RGBA", base.size, (0,0,0,0))
    layer.paste(Image.new("RGBA", base.size, (*bc_, 245)), (0,0), shirt_mask)

    shade = Image.new("RGBA", base.size, (0,0,0,0))
    spx   = shade.load()
    for x in range(W):
        dist  = abs(x - cx) / max(1, sw)
        alpha = int(max(0, min(42, (dist - 0.25) * 58)))
        if not alpha:
            continue
        for y in range(max(0, sy), min(H, int(sy + sw * 2.05)), 2):
            spx[x, y] = spx[x, y+1 if y+1 < H else y] = (0, 0, 0, alpha)
    layer = Image.alpha_composite(layer, Image.composite(shade, Image.new("RGBA", base.size, (0,0,0,0)), shirt_mask))

    fp = extract_print(garment, bc_)
    tw = int(sw * 0.62)
    th = int(tw * fp.height / max(1, fp.width))
    mh = int(sw * 0.88)
    if th > mh:
        th, tw = mh, int(mh * fp.width / max(1, fp.height))
    fp = fp.resize((max(24,tw), max(24,th)), Image.LANCZOS)
    layer.alpha_composite(fp, (int(cx - tw/2), int(cy + sw*0.38)))

    draw = ImageDraw.Draw(layer)
    cw, ch = sw * 0.34, sw * 0.15
    trim = tuple(max(0, min(255, int(c*0.72))) for c in bc_)
    draw.arc((cx-cw/2, cy-ch*0.35, cx+cw/2, cy+ch*0.70),
             start=180, end=360, fill=(*trim, 235), width=max(2, int(sw*0.026)))

    return Image.alpha_composite(base, layer).convert("RGB")

# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(value) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        for key in ("background", "path", "name"):
            v = value.get(key)
            if v is not None:
                return load_image(v) if key == "background" else Image.open(v)
    if isinstance(value, str):
        return Image.open(value)
    return None

# ── Runtime / model loading ───────────────────────────────────────────────────
#
# Loading is split into two phases so that the slow, network-bound parts don't
# happen on the first user's request (and don't count against the GPU
# `duration` budget of @spaces.GPU):
#
#   1. prepare_assets()  - CPU only. Downloads/locates the model bundle and
#                           resolves on-disk paths. Safe to run at import time,
#                           before any GPU has been allocated.
#
#   2. build_pipeline()  - GPU. Constructs the actual CatVTON pipeline (and
#                           AutoMasker). Must run inside @spaces.GPU on ZeroGPU
#                           Spaces. Result is cached in `RUNTIME`.
#
# get_runtime() returns the cached pipeline, building it on first call.
# A background warm-up thread calls get_runtime() (via a @spaces.GPU function)
# right at startup so that, in the common case, it's already built by the time
# a real user clicks "Run".

_PREP_LOCK = threading.Lock()
_PREP = None  # cached output of prepare_assets()

RUNTIME = None
_RUNTIME_LOCK = threading.Lock()


def normalize_category(text: str) -> str:
    t = (text or "").strip().lower()
    if "lower" in t:
        return "lower"
    if any(w in t for w in ("full","overall","dress","abaya","عباية","عبايات")):
        return "overall"
    return "upper"


def prepare_assets():
    """CPU-only step: locate or download the model bundle and resolve paths.

    Does not touch CUDA, so it can run at import time / Space startup,
    well before any GPU allocation is requested.
    """
    global _PREP
    if _PREP is not None:
        return _PREP

    with _PREP_LOCK:
        if _PREP is not None:
            return _PREP

        src = PROJECT_ROOT
        if not any(PROJECT_ROOT.rglob("model/pipeline.py")):
            # Try the local HF cache first (no network) before falling back
            # to a full download. On a warm container this can avoid most of
            # the cold-start cost entirely.
            try:
                src = Path(snapshot_download(
                    repo_id=MODEL_REPO_ID,
                    repo_type="model",
                    token=os.getenv("HF_TOKEN") or None,
                    local_files_only=True,
                ))
                print(f"Using cached model bundle for '{MODEL_REPO_ID}'.")
            except Exception:
                print(f"Downloading model bundle '{MODEL_REPO_ID}' (this may take a while)...")
                src = Path(snapshot_download(
                    repo_id=MODEL_REPO_ID,
                    repo_type="model",
                    token=os.getenv("HF_TOKEN") or None,
                    max_workers=8,
                ))
                print("Model bundle download complete.")

        bundle    = find_bundle(src)
        base_path = Path(BASE_MODEL_PATH_RAW).resolve() if BASE_MODEL_PATH_RAW else find_dir(src, "base_model")
        resume    = Path(RESUME_PATH_RAW).resolve()     if RESUME_PATH_RAW     else find_dir(src, "catvton_weights")

        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        if str(bundle) not in sys.path:
            sys.path.insert(0, str(bundle))

        if not base_path.exists():
            raise FileNotFoundError(f"Base model not found: {base_path}")
        if not resume.exists():
            raise FileNotFoundError(f"CatVTON weights not found: {resume}")

        _PREP = {"src": src, "bundle": bundle, "base_path": base_path, "resume": resume}
        return _PREP


def build_pipeline():
    """GPU step: instantiate the CatVTON pipeline (and AutoMasker if available).

    Must be called from within a @spaces.GPU-decorated function on ZeroGPU
    Spaces, since it allocates CUDA memory. The result should be cached -
    see get_runtime().
    """
    prep = prepare_assets()
    base_path, resume = prep["base_path"], prep["resume"]

    import model.pipeline as pm
    from model.pipeline import CatVTONPipeline

    AutoMasker = None
    try:
        from model.cloth_masker import AutoMasker
    except Exception as e:
        print(f"AutoMasker unavailable, using fallback mask: {e}")

    # Redirect HF hub loaders to local paths
    _vae  = pm.AutoencoderKL.from_pretrained
    _unet = pm.UNet2DConditionModel.from_pretrained

    def local_vae(path, *a, **kw):
        if path == "stabilityai/sd-vae-ft-mse" and (base_path / "vae").exists():
            kw.pop("subfolder", None)
            kw.setdefault("low_cpu_mem_usage", True)
            kw.setdefault("use_safetensors", False)
            return _vae(str(base_path), *a, subfolder="vae", **kw)
        return _vae(path, *a, **kw)

    def local_unet(path, *a, **kw):
        if Path(path).resolve() == base_path:
            kw.setdefault("low_cpu_mem_usage", True)
            kw.setdefault("use_safetensors", False)
            return _unet(str(base_path), *a, **kw)
        return _unet(path, *a, **kw)

    pm.AutoencoderKL.from_pretrained        = local_vae
    pm.UNet2DConditionModel.from_pretrained = local_unet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    masker = None
    dp, sc = resume / "DensePose", resume / "SCHP"
    if AutoMasker and dp.exists() and sc.exists():
        masker = AutoMasker(densepose_ckpt=str(dp), schp_ckpt=str(sc), device=device)

    pipeline = CatVTONPipeline(
        base_ckpt=str(base_path),
        attn_ckpt=str(resume),
        attn_ckpt_version=ATTN_VERSION,
        weight_dtype=dtype,
        device=device,
        skip_safety_check=True,
    )

    return {"pipeline": pipeline, "masker": masker, "device": device,
            "width": 384, "height": 512, "guidance_scale": 2.0}


def get_runtime():
    """Return the cached pipeline runtime, building it on first use.

    Thread-safe: if two requests race on first use, only one builds the
    pipeline and the other waits for it.
    """
    global RUNTIME
    if RUNTIME is None:
        with _RUNTIME_LOCK:
            if RUNTIME is None:
                RUNTIME = build_pipeline()
    return RUNTIME


@spaces.GPU(duration=WARMUP_GPU_DURATION)
def _warmup():
    """Build the pipeline eagerly so the first real request doesn't pay for it."""
    get_runtime()
    return "ready"


def _background_warmup():
    try:
        print("Warming up model pipeline (this happens once at startup)...")
        _warmup()
        print("Model pipeline warm-up complete.")
    except Exception as e:
        # Not fatal - get_runtime() will simply build it lazily on first
        # request instead (with the inference-time GPU duration budget).
        print(f"Warm-up failed, will build pipeline lazily on first request: {e}")

# ── Inference ─────────────────────────────────────────────────────────────────

def _infer_duration(person_image, cloth_image, garment_description="Upper body clothing garment",
                     auto_mask=True, denoise_steps=30, seed=555, pose_landmarks_data="",
                     guidance_scale=2.5):
    """Scale the GPU time budget with the work actually requested.

    More denoise steps and/or a first-time pipeline build both take longer;
    give plenty of headroom so a high-quality run isn't killed mid-way.
    """
    steps = int(denoise_steps)
    base  = 45 if RUNTIME is not None else 240  # account for a possible cold build
    return min(360, base + steps * 5)


@spaces.GPU(duration=_infer_duration)
def run_tryon(person_image, cloth_image, garment_description="Upper body clothing garment",
              auto_mask=True, denoise_steps=30, seed=555, pose_landmarks_data="",
              guidance_scale=2.5):
    person = load_image(person_image)
    cloth  = load_image(cloth_image)
    if person is None or cloth is None:
        raise gr.Error("Please upload both a person image and a clothing image.")

    person   = person.convert("RGB")
    cloth    = cloth.convert("RGB")
    category = normalize_category(garment_description)
    rt       = get_runtime()

    if auto_mask and rt["masker"] is not None:
        mask = rt["masker"](person, mask_type=category)["mask"]
        if category == "upper" and pose_landmarks_data:
            mask = ImageChops.lighter(mask.convert("L"),
                                      build_mask(person, category, pose_landmarks_data).convert("L"))
    else:
        mask = build_mask(person, category, pose_landmarks_data)
    mask = protect_identity(mask, category, pose_landmarks_data)

    gen = torch.Generator(device=rt["device"]).manual_seed(int(seed))
    results = rt["pipeline"](
        person, cloth, mask,
        num_inference_steps=int(denoise_steps),
        guidance_scale=float(guidance_scale),
        height=rt["height"], width=rt["width"],
        generator=gen,
    )
    result = results[0]
    if is_bad_result(result, person, pose_landmarks_data):
        return render_fallback(person, cloth, pose_landmarks_data)
    return result

# ── Startup ───────────────────────────────────────────────────────────────────

# 1. Resolve/download model assets now (CPU only, no GPU needed) so this
#    cost is paid once during Space boot rather than during a user request.
try:
    prepare_assets()
except Exception as e:
    print(f"Asset preparation failed: {e}")

# 2. Kick off pipeline construction in the background. Using a thread keeps
#    the Gradio server itself responsive immediately, while the model
#    finishes loading shortly after the Space reports as "Running".
threading.Thread(target=_background_warmup, daemon=True).start()

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Virtual Fitting Room") as demo:
    gr.Markdown("# Virtual Fitting Room\nUpload a person photo and a garment, then hit **Run**.")
    gr.Markdown(
        "_Note: the model finishes loading shortly after the app starts. "
        "If your first try-on takes longer than usual, that's the one-time "
        "warm-up - later runs will be faster._"
    )

    with gr.Row():
        person_input = gr.ImageEditor(type="pil", label="Person Image", sources=["upload","webcam"])
        cloth_input  = gr.Image(type="pil", label="Clothing Image", sources=["upload"])

    garment_desc   = gr.Textbox(value="Upper body clothing garment", label="Garment Description")
    auto_mask      = gr.Checkbox(value=True, label="Auto Mask")
    denoise_steps  = gr.Slider(10, 50, value=30, step=1, label="Denoise Steps",
                                info="Higher = better quality, slower. 30-40 is a good balance, 50 for max quality.")
    guidance_scale = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="Guidance Scale",
                                info="Higher = follows the garment image more closely. 2.5-3.5 usually looks best.")
    seed           = gr.Number(value=555, precision=0, label="Seed")
    pose_landmarks = gr.Textbox(value="", visible=False, label="Pose Landmarks")

    run_btn     = gr.Button("Run Try-On", variant="primary")
    output_image = gr.Image(type="pil", label="Result")

    run_btn.click(
        fn=run_tryon,
        inputs=[person_input, cloth_input, garment_desc, auto_mask, denoise_steps, seed, pose_landmarks, guidance_scale],
        outputs=output_image,
        api_name="tryon",
    )

if __name__ == "__main__":
    demo.launch()
