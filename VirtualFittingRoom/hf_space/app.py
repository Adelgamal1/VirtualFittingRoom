import io
import json
import os
import sys
from pathlib import Path

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat


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


def mean_abs_difference(image_a: Image.Image, image_b: Image.Image, box) -> float:
    try:
        crop_a = image_a.crop(box).resize((64, 64)).convert("RGB")
        crop_b = image_b.crop(box).resize((64, 64)).convert("RGB")
        diff = ImageChops.difference(crop_a, crop_b)
        stat = ImageStat.Stat(diff)
        return sum(stat.mean) / 3.0
    except Exception:
        return 0.0


def is_unstable_tryon_result(result: Image.Image, person_image: Image.Image, pose_landmarks_data=None) -> bool:
    result = result.convert("RGB")
    person_image = person_image.convert("RGB")
    if result.size != person_image.size:
        result = result.resize(person_image.size, Image.LANCZOS)

    width, height = person_image.size
    pose = read_pose_landmarks(pose_landmarks_data, person_image.size)
    if pose:
        shoulder_width = pose["shoulder_width"]
        collar_y = int(pose["collar_y"])
        protected_bottom = max(1, int(collar_y - shoulder_width * 0.12))
    else:
        protected_bottom = max(1, int(height * 0.22))

    head_diff = mean_abs_difference(result, person_image, (0, 0, width, protected_bottom))
    if head_diff > 34:
        return True

    # A very large change in the whole upper frame usually means the diffusion model
    # repeated the garment as a collage instead of fitting it on the torso.
    upper_diff = mean_abs_difference(result, person_image, (0, 0, width, int(height * 0.72)))
    return upper_diff > 82


def garment_foreground_bbox(image: Image.Image):
    image = image.convert("RGB")
    width, height = image.size
    min_x, min_y = width, height
    max_x, max_y = -1, -1
    pixels = image.load()
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            r, g, b = pixels[x, y]
            if r > 238 and g > 238 and b > 238:
                continue
            if max(r, g, b) - min(r, g, b) < 4 and max(r, g, b) > 230:
                continue
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    if max_x < min_x or max_y < min_y:
        return (0, 0, width, height)
    return (
        max(0, min_x - 4),
        max(0, min_y - 4),
        min(width, max_x + 5),
        min(height, max_y + 5),
    )


def estimate_garment_base_color(garment: Image.Image):
    image = garment.convert("RGB").resize((180, 180))
    colors = []
    dark_colors = []
    for r, g, b in image.getdata():
        if r > 238 and g > 238 and b > 238:
            continue
        brightness = (r + g + b) / 3
        colors.append((r, g, b))
        if brightness < 95:
            dark_colors.append((r, g, b))

    sample = dark_colors if len(dark_colors) > 30 else colors
    if not sample:
        return (24, 24, 27)

    sample = sorted(sample, key=lambda c: (c[0] + c[1] + c[2]))
    mid = sample[len(sample) // 2]
    return tuple(max(0, min(255, int(v))) for v in mid)


def extract_front_print(garment: Image.Image, base_color):
    image = garment.convert("RGB")
    fg = garment_foreground_bbox(image)
    crop = image.crop(fg)
    width, height = crop.size
    pixels = crop.load()
    base_r, base_g, base_b = base_color
    min_x, min_y = width, height
    max_x, max_y = -1, -1

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r > 242 and g > 242 and b > 242:
                continue
            distance = abs(r - base_r) + abs(g - base_g) + abs(b - base_b)
            if distance < 70:
                continue
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    if max_x < min_x or max_y < min_y:
        min_x = int(width * 0.24)
        max_x = int(width * 0.76)
        min_y = int(height * 0.22)
        max_y = int(height * 0.72)
    else:
        pad_x = int((max_x - min_x + 1) * 0.08)
        pad_y = int((max_y - min_y + 1) * 0.08)
        min_x = max(0, min_x - pad_x)
        max_x = min(width - 1, max_x + pad_x)
        min_y = max(0, min_y - pad_y)
        max_y = min(height - 1, max_y + pad_y)

    print_crop = crop.crop((min_x, min_y, max_x + 1, max_y + 1)).convert("RGBA")
    alpha = Image.new("L", print_crop.size, 0)
    alpha_pixels = alpha.load()
    print_pixels = print_crop.load()
    for y in range(print_crop.height):
        for x in range(print_crop.width):
            r, g, b, _ = print_pixels[x, y]
            if r > 245 and g > 245 and b > 245:
                continue
            alpha_pixels[x, y] = 255
    print_crop.putalpha(alpha.filter(ImageFilter.GaussianBlur(radius=0.4)))
    return print_crop


def render_controlled_upper_tryon(person_image: Image.Image, garment_image: Image.Image, pose_landmarks_data=None) -> Image.Image:
    person = person_image.convert("RGBA")
    width, height = person.size
    pose = read_pose_landmarks(pose_landmarks_data, person.size)
    if not pose:
        return person_image.convert("RGB")

    shoulder_width = pose["shoulder_width"]
    center_x = int((pose["shoulder_center"][0] * 0.66) + (pose["hip_center"][0] * 0.34))
    collar_y = int(pose["collar_y"])
    shoulder_y = int(pose["shoulder_center"][1] - shoulder_width * 0.075)
    base_color = estimate_garment_base_color(garment_image)

    shirt_mask = build_fallback_mask(person_image, "upper", pose_landmarks_data)
    shirt_mask = protect_identity_regions(shirt_mask, "upper", pose_landmarks_data)
    shirt_mask = shirt_mask.filter(ImageFilter.GaussianBlur(radius=0.9))

    shirt_layer = Image.new("RGBA", person.size, (0, 0, 0, 0))
    shirt_fill = Image.new("RGBA", person.size, (*base_color, 245))
    shirt_layer.paste(shirt_fill, (0, 0), shirt_mask)

    shade = Image.new("RGBA", person.size, (0, 0, 0, 0))
    shade_pixels = shade.load()
    for x in range(width):
        distance = abs(x - center_x) / max(1, shoulder_width)
        alpha = int(max(0, min(42, (distance - 0.25) * 58)))
        if alpha <= 0:
            continue
        for y in range(max(0, shoulder_y), min(height, int(shoulder_y + shoulder_width * 2.05)), 2):
            shade_pixels[x, y] = (0, 0, 0, alpha)
            if y + 1 < height:
                shade_pixels[x, y + 1] = (0, 0, 0, alpha)
    shirt_layer = Image.alpha_composite(shirt_layer, Image.composite(shade, Image.new("RGBA", person.size, (0, 0, 0, 0)), shirt_mask))

    front_print = extract_front_print(garment_image, base_color)
    target_w = int(shoulder_width * 0.62)
    target_h = int(target_w * front_print.height / max(1, front_print.width))
    max_h = int(shoulder_width * 0.88)
    if target_h > max_h:
        target_h = max_h
        target_w = int(target_h * front_print.width / max(1, front_print.height))
    target_w = max(24, target_w)
    target_h = max(24, target_h)
    front_print = front_print.resize((target_w, target_h), Image.LANCZOS)
    print_x = int(center_x - target_w / 2)
    print_y = int(collar_y + shoulder_width * 0.38)
    shirt_layer.alpha_composite(front_print, (print_x, print_y))

    draw = ImageDraw.Draw(shirt_layer)
    collar_w = shoulder_width * 0.34
    collar_h = shoulder_width * 0.15
    collar_box = (
        center_x - collar_w / 2,
        collar_y - collar_h * 0.35,
        center_x + collar_w / 2,
        collar_y + collar_h * 0.70,
    )
    trim = tuple(max(0, min(255, int(channel * 0.72))) for channel in base_color)
    draw.arc(collar_box, start=180, end=360, fill=(*trim, 235), width=max(2, int(shoulder_width * 0.026)))

    result = Image.alpha_composite(person, shirt_layer)
    return result.convert("RGB")


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
        if category == "upper" and pose_landmarks_data:
            pose_mask = build_fallback_mask(person_image, category, pose_landmarks_data)
            mask = ImageChops.lighter(mask.convert("L"), pose_mask.convert("L"))
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
    result = results[0]
    if is_unstable_tryon_result(result, person_image, pose_landmarks_data):
        return render_controlled_upper_tryon(person_image, cloth_image, pose_landmarks_data)
    return result


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
