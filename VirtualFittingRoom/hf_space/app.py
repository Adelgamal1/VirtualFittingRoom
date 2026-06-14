import io
import json
import os
import sys
from pathlib import Path

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFilter


SPACE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("CATVTON_PROJECT_ROOT", SPACE_ROOT))
BASE_MODEL_PATH_RAW = os.getenv("CATVTON_BASE_MODEL_PATH", "").strip()
RESUME_PATH_RAW = os.getenv("CATVTON_RESUME_PATH", "").strip()
ATTN_VERSION = os.getenv("CATVTON_ATTN_VERSION", "mix")
MODEL_REPO_ID = os.getenv("CATVTON_MODEL_REPO_ID", "Adelgamal1/virtual-fitting-room-model")


def find_project_bundle(project_root: Path) -> Path:
    if (project_root / "model" / "pipeline.py").exists():
        return project_root

    for candidate in project_root.iterdir():
        if candidate.is_dir() and (candidate / "model" / "pipeline.py").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a model bundle under {project_root} containing model/pipeline.py."
    )


def find_nested_directory(project_root: Path, directory_name: str) -> Path:
    direct = project_root / directory_name
    if direct.is_dir():
        return direct

    matches = sorted(
        [candidate for candidate in project_root.rglob(directory_name) if candidate.is_dir()],
        key=lambda path: str(path),
    )
    if not matches:
        raise FileNotFoundError(f"Required directory '{directory_name}' was not found under {project_root}.")
    return matches[-1]


def read_pose_landmarks(pose_landmarks_data, image_size):
    if not pose_landmarks_data:
        return None

    try:
        payload = json.loads(pose_landmarks_data)
        landmarks = payload.get("landmarks") or []
        width, height = image_size

        def read_point(index, min_visibility=0.24):
            if index >= len(landmarks):
                return None
            item = landmarks[index] or {}
            visibility = float(item.get("visibility", 1.0))
            if visibility < min_visibility:
                return None

            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
            if x <= 1.5 and y <= 1.5:
                x *= width
                y *= height
            return (x, y, visibility)

        shoulder_a = read_point(11, 0.34)
        shoulder_b = read_point(12, 0.34)
        hip_a = read_point(23, 0.22)
        hip_b = read_point(24, 0.22)
        if not shoulder_a or not shoulder_b or not hip_a or not hip_b:
            return None

        left_shoulder, right_shoulder = sorted([shoulder_a, shoulder_b], key=lambda point: point[0])
        left_hip, right_hip = sorted([hip_a, hip_b], key=lambda point: point[0])
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        if shoulder_width < width * 0.08 or shoulder_width > width * 0.78:
            return None

        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
        )
        hip_center = (
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2,
        )
        torso_height = hip_center[1] - shoulder_center[1]
        if torso_height < height * 0.15:
            return None

        return {
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "left_hip": left_hip,
            "right_hip": right_hip,
            "shoulder_center": shoulder_center,
            "hip_center": hip_center,
            "shoulder_width": shoulder_width,
            "torso_height": torso_height,
            "collar_y": shoulder_center[1] - (shoulder_width * 0.09),
            "neck_y": shoulder_center[1] - (shoulder_width * 0.18),
        }
    except Exception:
        return None


def build_fallback_mask(image: Image.Image, category: str, pose_landmarks_data=None) -> Image.Image:
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    pose = read_pose_landmarks(pose_landmarks_data, image.size)

    if category == "upper":
        if pose:
            shoulder_width = pose["shoulder_width"]
            shoulder_y = int(pose["shoulder_center"][1] - (shoulder_width * 0.075))
            collar_y = int(pose["collar_y"])
            waist_y = pose["shoulder_center"][1] + (pose["torso_height"] * 0.58)
            hem_y = int(min(height * 0.84, waist_y + (shoulder_width * 0.08)))
            center = int((pose["shoulder_center"][0] * 0.64) + (pose["hip_center"][0] * 0.36))
            left_shoulder_x = int(pose["left_shoulder"][0])
            right_shoulder_x = int(pose["right_shoulder"][0])
            left = int(max(0, left_shoulder_x - (shoulder_width * 0.46)))
            right = int(min(width, right_shoulder_x + (shoulder_width * 0.46)))
            bottom_half = int(max(shoulder_width * 0.50, abs(pose["right_hip"][0] - pose["left_hip"][0]) * 0.62))

            shirt_shape = [
                (left, int(shoulder_y + shoulder_width * 0.16)),
                (int(left_shoulder_x - shoulder_width * 0.06), collar_y),
                (int(center - shoulder_width * 0.22), int(collar_y - shoulder_width * 0.035)),
                (int(center + shoulder_width * 0.22), int(collar_y - shoulder_width * 0.035)),
                (int(right_shoulder_x + shoulder_width * 0.06), collar_y),
                (right, int(shoulder_y + shoulder_width * 0.16)),
                (int(center + bottom_half), hem_y),
                (int(center - bottom_half), hem_y),
            ]
            draw.polygon(shirt_shape, fill=255)

            neck_cutout = (
                int(center - shoulder_width * 0.155),
                int(collar_y - shoulder_width * 0.075),
                int(center + shoulder_width * 0.155),
                int(collar_y + shoulder_width * 0.070),
            )
            draw.ellipse(neck_cutout, fill=0)
        else:
            shoulder_y = int(height * 0.20)
            hem_y = int(height * 0.64)
            left = int(width * 0.14)
            right = int(width * 0.86)
            center = width // 2

            shirt_shape = [
                (center, shoulder_y),
                (right, int(height * 0.23)),
                (int(width * 0.80), hem_y),
                (int(width * 0.20), hem_y),
                (left, int(height * 0.23)),
            ]
            draw.polygon(shirt_shape, fill=255)

            # Preserve face and neck when DensePose/SCHP are unavailable.
            neck_cutout = (
                int(width * 0.40),
                int(height * 0.13),
                int(width * 0.60),
                int(height * 0.27),
            )
            draw.ellipse(neck_cutout, fill=0)
    elif category == "lower":
        box = (int(width * 0.22), int(height * 0.42), int(width * 0.78), int(height * 0.96))
        draw.rounded_rectangle(box, radius=max(12, width // 16), fill=255)
    else:
        box = (int(width * 0.16), int(height * 0.06), int(width * 0.84), int(height * 0.96))
        draw.rounded_rectangle(box, radius=max(12, width // 16), fill=255)
    if category == "upper":
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1.2))
    return mask


def normalize_category(value: str) -> str:
    text = (value or "").strip().lower()
    if "lower" in text:
        return "lower"
    if "full" in text or "overall" in text or "dress" in text or "abaya" in text or "عباية" in text or "عبايات" in text:
        return "overall"
    return "upper"


def protect_identity_regions(mask: Image.Image, category: str, pose_landmarks_data=None) -> Image.Image:
    if category != "upper":
        return mask

    mask = mask.convert("L")
    width, height = mask.size
    protected = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(protected)
    pose = read_pose_landmarks(pose_landmarks_data, mask.size)

    if pose:
        shoulder_width = pose["shoulder_width"]
        center_x = int(pose["shoulder_center"][0])
        collar_y = int(pose["collar_y"])
        draw.rectangle((0, 0, width, max(0, int(collar_y - shoulder_width * 0.18))), fill=255)
        draw.ellipse(
            (
                int(center_x - shoulder_width * 0.16),
                int(collar_y - shoulder_width * 0.16),
                int(center_x + shoulder_width * 0.16),
                int(collar_y + shoulder_width * 0.07),
            ),
            fill=255,
        )
        mask.paste(0, mask=protected)
        return mask

    draw.rectangle((0, 0, width, int(height * 0.15)), fill=255)
    draw.ellipse(
        (
            int(width * 0.37),
            int(height * 0.10),
            int(width * 0.63),
            int(height * 0.30),
        ),
        fill=255,
    )

    mask.paste(0, mask=protected)
    return mask


def load_image_value(value) -> Image.Image | None:
    if value is None:
        return None

    if isinstance(value, Image.Image):
        return value

    if isinstance(value, dict):
        if "background" in value and value["background"] is not None:
            return load_image_value(value["background"])
        if "path" in value and value["path"] is not None:
            return Image.open(value["path"])
        if "name" in value and value["name"] is not None:
            return Image.open(value["name"])

    if isinstance(value, str):
        return Image.open(value)

    return None


def load_runtime():
    source_root = PROJECT_ROOT
    if not any(PROJECT_ROOT.rglob("model/pipeline.py")):
        snapshot_root = snapshot_download(
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            token=os.getenv("HF_TOKEN") or None,
        )
        source_root = Path(snapshot_root)

    bundle_root = find_project_bundle(source_root)
    base_model_path = Path(BASE_MODEL_PATH_RAW).resolve() if BASE_MODEL_PATH_RAW else find_nested_directory(source_root, "base_model")
    resume_path = Path(RESUME_PATH_RAW).resolve() if RESUME_PATH_RAW else find_nested_directory(source_root, "catvton_weights")
    densepose_root = resume_path / "DensePose"
    schp_root = resume_path / "SCHP"

    sys.path.insert(0, str(source_root))
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
        "width": 384,
        "height": 512,
        "steps": 8,
        "guidance_scale": 2.0,
    }


RUNTIME = None


def get_runtime():
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = load_runtime()
    return RUNTIME


@spaces.GPU(duration=180)
def run_tryon(
    person_image,
    cloth_image,
    garment_description="Upper body clothing garment",
    auto_mask=True,
    auto_crop=False,
    denoise_steps=20,
    seed=555,
    pose_landmarks_data="",
):
    person_image = load_image_value(person_image)
    cloth_image = load_image_value(cloth_image)

    if person_image is None or cloth_image is None:
        raise gr.Error("Please upload both the person image and the clothing image.")

    person_image = person_image.convert("RGB")
    cloth_image = cloth_image.convert("RGB")
    category = normalize_category(garment_description)

    runtime = get_runtime()

    if auto_mask and runtime["masker"] is not None:
        mask_result = runtime["masker"](person_image, mask_type=category)
        mask = mask_result["mask"]
    else:
        mask = build_fallback_mask(person_image, category, pose_landmarks_data)
    mask = protect_identity_regions(mask, category, pose_landmarks_data)

    generator = torch.Generator(device=runtime["device"])
    generator.manual_seed(int(seed))

    results = runtime["pipeline"](
        person_image,
        cloth_image,
        mask,
        num_inference_steps=int(denoise_steps),
        guidance_scale=runtime["guidance_scale"],
        height=runtime["height"],
        width=runtime["width"],
        generator=generator,
    )
    return results[0]


with gr.Blocks(title="Virtual Fitting Room") as demo:
    gr.Markdown("# Virtual Fitting Room")
    gr.Markdown(
        "Upload a person image or capture one from the webcam, then upload a clothing image "
        "and choose upper, lower, or overall try-on."
    )

    with gr.Row():
        person_input = gr.ImageEditor(
            type="pil",
            label="Person Image",
            sources=["upload", "webcam"],
        )
        cloth_input = gr.Image(
            type="pil",
            label="Clothing Image",
            sources=["upload"],
        )

    garment_description_input = gr.Textbox(
        value="Upper body clothing garment",
        label="Garment Description",
    )

    with gr.Row():
        auto_mask_input = gr.Checkbox(value=True, label="Auto Mask")
        auto_crop_input = gr.Checkbox(value=False, label="Auto Crop")

    with gr.Row():
        denoise_steps_input = gr.Slider(10, 50, value=20, step=1, label="Denoise Steps")
        seed_input = gr.Number(value=555, precision=0, label="Seed")
    pose_landmarks_input = gr.Textbox(value="", visible=False, label="Pose Landmarks")

    run_button = gr.Button("Run Try-On", variant="primary")
    output_image = gr.Image(type="pil", label="Result")

    run_button.click(
        fn=run_tryon,
        inputs=[
            person_input,
            cloth_input,
            garment_description_input,
            auto_mask_input,
            auto_crop_input,
            denoise_steps_input,
            seed_input,
            pose_landmarks_input,
        ],
        outputs=output_image,
        api_name="tryon",
    )


if __name__ == "__main__":
    demo.launch()
