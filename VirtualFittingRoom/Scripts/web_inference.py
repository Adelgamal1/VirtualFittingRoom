import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from PIL import ImageDraw


def find_project_bundle(project_root: Path) -> Path:
    for candidate in project_root.iterdir():
        if candidate.is_dir() and (candidate / "model" / "pipeline.py").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a model bundle under {project_root} containing model/pipeline.py."
    )


def find_nested_directory(project_root: Path, directory_name: str) -> Path:
    matches = sorted(
        [
            candidate
            for candidate in project_root.rglob(directory_name)
            if candidate.is_dir()
        ],
        key=lambda path: str(path),
    )
    if not matches:
        raise FileNotFoundError(f"Required directory '{directory_name}' was not found under {project_root}.")
    return matches[-1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-image web inference for Virtual Fitting Room.")
    parser.add_argument("--person", required=True, help="Path to the user/person image.")
    parser.add_argument("--cloth", required=True, help="Path to the clothing image.")
    parser.add_argument("--category", required=True, choices=["upper", "lower", "overall"], help="Garment area.")
    parser.add_argument("--output", required=True, help="Where to save the generated result.")
    parser.add_argument("--project-root", default=os.getcwd(), help="Root folder of the CatVTON project.")
    parser.add_argument("--base-model-path", default="")
    parser.add_argument("--resume-path", default="")
    parser.add_argument("--attn-version", default="mix", choices=["mix", "vitonhd", "dresscode"])
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    return parser.parse_args()


def build_fallback_mask(image: Image.Image, category: str) -> Image.Image:
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if category == "upper":
        box = (
            int(width * 0.18),
            int(height * 0.08),
            int(width * 0.82),
            int(height * 0.58),
        )
    elif category == "lower":
        box = (
            int(width * 0.22),
            int(height * 0.42),
            int(width * 0.78),
            int(height * 0.96),
        )
    else:
        box = (
            int(width * 0.16),
            int(height * 0.06),
            int(width * 0.84),
            int(height * 0.96),
        )

    draw.rounded_rectangle(box, radius=max(12, width // 16), fill=255)
    return mask


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    bundle_root = find_project_bundle(project_root)
    base_model_path = Path(args.base_model_path).resolve() if args.base_model_path else find_nested_directory(project_root, "base_model")
    resume_path = Path(args.resume_path).resolve() if args.resume_path else find_nested_directory(project_root, "catvton_weights")
    densepose_root = resume_path / "DensePose"
    schp_root = resume_path / "SCHP"

    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(bundle_root))

    import model.pipeline as pipeline_module
    from model.pipeline import CatVTONPipeline
    try:
        from model.cloth_masker import AutoMasker
    except Exception as ex:
        AutoMasker = None
        print(f"AutoMasker import failed, using fallback mask: {ex}")

    if not densepose_root.exists():
        raise FileNotFoundError(f"DensePose directory was not found at '{densepose_root}'.")
    if not schp_root.exists():
        raise FileNotFoundError(f"SCHP directory was not found at '{schp_root}'.")
    if not (base_model_path / "model_index.json").exists():
        raise FileNotFoundError(f"Base model path '{base_model_path}' does not look valid.")
    if not (resume_path / "mix-48k-1024").exists():
        raise FileNotFoundError(f"CATVTON weights path '{resume_path}' does not contain attention checkpoints.")

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
            if (base_model_path / "unet" / "diffusion_pytorch_model.safetensors").exists():
                loader_kwargs.setdefault("use_safetensors", True)
            else:
                loader_kwargs.setdefault("use_safetensors", False)
            return original_unet_loader(str(base_model_path), *loader_args, **loader_kwargs)
        return original_unet_loader(model_name_or_path, *loader_args, **loader_kwargs)

    pipeline_module.AutoencoderKL.from_pretrained = load_local_vae
    pipeline_module.UNet2DConditionModel.from_pretrained = load_local_unet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    person_image = Image.open(args.person).convert("RGB")
    cloth_image = Image.open(args.cloth).convert("RGB")

    if AutoMasker is not None:
        masker = AutoMasker(
            densepose_ckpt=str(densepose_root),
            schp_ckpt=str(schp_root),
            device=device,
        )
        mask_result = masker(person_image, mask_type=args.category)
        mask = mask_result["mask"]
    else:
        mask = build_fallback_mask(person_image, args.category)

    pipeline = CatVTONPipeline(
        base_ckpt=str(base_model_path),
        attn_ckpt=str(resume_path),
        attn_ckpt_version=args.attn_version,
        weight_dtype=weight_dtype,
        device=device,
        skip_safety_check=True,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(555)

    results = pipeline(
        person_image,
        cloth_image,
        mask,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results[0].save(output_path)


if __name__ == "__main__":
    main()
