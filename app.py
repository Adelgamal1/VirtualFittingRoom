import io
import os
import sys
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageDraw


SPACE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("CATVTON_PROJECT_ROOT", SPACE_ROOT))
BASE_MODEL_PATH = Path(os.getenv("CATVTON_BASE_MODEL_PATH", ""))
RESUME_PATH = Path(os.getenv("CATVTON_RESUME_PATH", ""))
ATTN_VERSION = os.getenv("CATVTON_ATTN_VERSION", "mix")


def find_project_bundle(project_root: Path) -> Path:
    for candidate in project_root.iterdir():
        if candidate.is_dir() and (candidate / "model" / "pipeline.py").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a model bundle under {project_root} containing model/pipeline.py."
    )


def find_nested_directory(project_root: Path, directory_name: str) -> Path:
    matches = sorted(
        [candidate for candidate in project_root.rglob(directory_name) if candidate.is_dir()],
        key=lambda path: str(path),
    )
    if not matches:
        raise FileNotFoundError(f"Required directory '{directory_name}' was not found under {project_root}.")
    return matches[-1]


def build_fallback_mask(image: Image.Image, category: str) -> Image.Image:
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if category == "upper":
        box = (int(width * 0.18), int(height * 0.08), int(width * 0.82), int(height * 0.58))
    elif category == "lower":
        box = (int(width * 0.22), int(height * 0.42), int(width * 0.78), int(height * 0.96))
    else:
        box = (int(width * 0.16), int(height * 0.06), int(width * 0.84), int(height * 0.96))

    draw.rounded_rectangle(box, radius=max(12, width // 16), fill=255)
    return mask


def load_runtime():
    bundle_root = find_project_bundle(PROJECT_ROOT)
    base_model_path = BASE_MODEL_PATH.resolve() if str(BASE_MODEL_PATH) else find_nested_directory(PROJECT_ROOT, "base_model")
    resume_path = RESUME_PATH.resolve() if str(RESUME_PATH) else find_nested_directory(PROJECT_ROOT, "catvton_weights")
    densepose_root = resume_path / "DensePose"
    schp_root = resume_path / "SCHP"

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(bundle_root))

    import model.pipeline as pipeline_module
    from model.pipeline import CatVTONPipeline

    try:
        from model.cloth_masker import AutoMasker
    except Exception as ex:
        AutoMasker = None
        print(f"AutoMasker import failed, using fallback mask: {ex}")

    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model path was not found: {base_model_path}")
    if not resume_path.exists():
        raise FileNotFoundError(f"CATVTON weights path was not found: {resume_path}")

    original_vae_loader = pipeline_module.AutoencoderKL.from_pretrained
    original_unet_loader = pipeline_module.UNet2DConditionModel.from_pretrained

    def load_local_vae(model_name_or_path, *loader_args, **loader_kwargs):
        if model_name_or_path == "stabilityai/sd-vae-ft-mse" and (base_model_path / "vae").exists():
            loader_kwargs.pop("subfolder", None)
            loader_kwargs.setdefault("low_cpu_mem_usage", True)
            loader_kwargs.setdefault("use_safetensors", False)
            return original_vae_loader(str(base_model_path), *loader_args, subfolder="vae", **loader_kwargs)
        return original_vae_loader(model_name_or_path, *loader_args, **loader_kwargs)

    def load_local_unet(model_name_or_path, *loader_args, **loader_kwargs):
        resolved_path = Path(model_name_or_path).resolve() if os.path.exists(model_name_or_path) else None
        if resolved_path == base_model_path:
            loader_kwargs.setdefault("low_cpu_mem_usage", True)
            loader_kwargs.setdefault("use_safetensors", False)
            return original_unet_loader(str(base_model_path), *loader_args, **loader_kwargs)
        return original_unet_loader(model_name_or_path, *loader_args, **loader_kwargs)

    pipeline_module.AutoencoderKL.from_pretrained = load_local_vae
    pipeline_module.UNet2DConditionModel.from_pretrained = load_local_unet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    masker = None
    if AutoMasker is not None and densepose_root.exists() and schp_root.exists():
        masker = AutoMasker(
            densepose_ckpt=str(densepose_root),
            schp_ckpt=str(schp_root),
            device=device,
        )

    pipeline = CatVTONPipeline(
        base_ckpt=str(base_model_path),
        attn_ckpt=str(resume_path),
        attn_ckpt_version=ATTN_VERSION,
        weight_dtype=weight_dtype,
        device=device,
        skip_safety_check=True,
    )

    return {
        "pipeline": pipeline,
        "masker": masker,
        "device": device,
        "width": 768,
        "height": 1024,
        "steps": 20,
        "guidance_scale": 2.5,
    }


RUNTIME = None


def get_runtime():
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = load_runtime()
    return RUNTIME


def run_tryon(person_image, cloth_image, category):
    if person_image is None or cloth_image is None:
        raise gr.Error("Please upload both the person image and the clothing image.")

    runtime = get_runtime()
    person_image = person_image.convert("RGB")
    cloth_image = cloth_image.convert("RGB")

    if runtime["masker"] is not None:
        mask_result = runtime["masker"](person_image, mask_type=category)
        mask = mask_result["mask"]
    else:
        mask = build_fallback_mask(person_image, category)

    generator = torch.Generator(device=runtime["device"])
    generator.manual_seed(555)

    results = runtime["pipeline"](
        person_image,
        cloth_image,
        mask,
        num_inference_steps=runtime["steps"],
        guidance_scale=runtime["guidance_scale"],
        height=runtime["height"],
        width=runtime["width"],
        generator=generator,
    )
    return results[0]


with gr.Blocks(title="Virtual Fitting Room") as demo:
    gr.Markdown("# Virtual Fitting Room")
    gr.Markdown("Upload a person image and a clothing image, then choose upper, lower, or overall try-on.")

    with gr.Row():
        person_input = gr.Image(type="pil", label="Person Image")
        cloth_input = gr.Image(type="pil", label="Clothing Image")

    category_input = gr.Radio(
        choices=["upper", "lower", "overall"],
        value="upper",
        label="Garment Area",
    )

    run_button = gr.Button("Run Try-On", variant="primary")
    output_image = gr.Image(type="pil", label="Result")

    run_button.click(
        fn=run_tryon,
        inputs=[person_input, cloth_input, category_input],
        outputs=output_image,
        api_name="tryon",
    )


if __name__ == "__main__":
    demo.launch()
