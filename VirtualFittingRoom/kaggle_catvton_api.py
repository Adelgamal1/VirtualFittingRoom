import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFilter

REPO_DIR = Path("/kaggle/working/CatVTON")
WIDTH = 768
HEIGHT = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))

from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, repaint_result, resize_and_crop, resize_and_padding

try:
    from model.cloth_masker import AutoMasker
except Exception as exc:
    print(f"AutoMasker unavailable, using fallback mask: {exc}")
    AutoMasker = None


def build_runtime():
    repo_path = snapshot_download(repo_id="zhengchong/CatVTON")
    pipeline = CatVTONPipeline(
        base_ckpt="runwayml/stable-diffusion-inpainting",
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype("fp16") if DEVICE == "cuda" else init_weight_dtype("no"),
        use_tf32=True,
        device=DEVICE,
        skip_safety_check=True,
    )
    masker = None
    if AutoMasker is not None:
        densepose_path = Path(repo_path) / "DensePose"
        schp_path = Path(repo_path) / "SCHP"
        if densepose_path.exists() and schp_path.exists():
            try:
                masker = AutoMasker(
                    densepose_ckpt=str(densepose_path),
                    schp_ckpt=str(schp_path),
                    device=DEVICE,
                )
            except Exception as exc:
                print(f"AutoMasker failed, using fallback mask: {exc}")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"pipeline": pipeline, "masker": masker}


RUNTIME = build_runtime()
PIPELINE = RUNTIME["pipeline"]
api = FastAPI(title="CatVTON Kaggle API")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def detect_face(image: Image.Image):
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def read_pose(data: str, size):
    if not data:
        return None
    try:
        landmarks = json.loads(data).get("landmarks") or []
        width, height = size

        def point(index, min_visibility=0.24):
            if index >= len(landmarks):
                return None
            landmark = landmarks[index] or {}
            visibility = float(landmark.get("visibility", 1.0))
            if visibility < min_visibility:
                return None
            x, y = float(landmark.get("x", 0)), float(landmark.get("y", 0))
            if x <= 1.5 and y <= 1.5:
                x, y = x * width, y * height
            return (x, y, visibility)

        left_shoulder, right_shoulder = point(11, 0.34), point(12, 0.34)
        left_hip, right_hip = point(23, 0.22), point(24, 0.22)
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None

        left_shoulder, right_shoulder = sorted([left_shoulder, right_shoulder], key=lambda p: p[0])
        left_hip, right_hip = sorted([left_hip, right_hip], key=lambda p: p[0])
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        if not (width * 0.08 < shoulder_width < width * 0.78):
            return None

        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
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
            "collar_y": shoulder_center[1] - shoulder_width * 0.09,
        }
    except Exception:
        return None


def controlled_upper_mask(image: Image.Image, pose_data: str) -> Image.Image:
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    pose = read_pose(pose_data, image.size)

    if not pose:
        return mask

    shoulder_width = pose["shoulder_width"]
    shoulder_y = int(pose["shoulder_center"][1] - shoulder_width * 0.075)
    collar_y = int(pose["collar_y"])
    hem_y = int(min(height * 0.84, pose["shoulder_center"][1] + pose["torso_height"] * 0.58 + shoulder_width * 0.08))
    center_x = int(pose["shoulder_center"][0] * 0.64 + pose["hip_center"][0] * 0.36)
    left_shoulder_x = int(pose["left_shoulder"][0])
    right_shoulder_x = int(pose["right_shoulder"][0])
    left_x = int(max(0, left_shoulder_x - shoulder_width * 0.46))
    right_x = int(min(width, right_shoulder_x + shoulder_width * 0.46))
    body_half = int(max(shoulder_width * 0.50, abs(pose["right_hip"][0] - pose["left_hip"][0]) * 0.62))

    draw.polygon(
        [
            (left_x, int(shoulder_y + shoulder_width * 0.16)),
            (int(left_shoulder_x - shoulder_width * 0.06), collar_y),
            (int(center_x - shoulder_width * 0.22), int(collar_y - shoulder_width * 0.035)),
            (int(center_x + shoulder_width * 0.22), int(collar_y - shoulder_width * 0.035)),
            (int(right_shoulder_x + shoulder_width * 0.06), collar_y),
            (right_x, int(shoulder_y + shoulder_width * 0.16)),
            (int(center_x + body_half), hem_y),
            (int(center_x - body_half), hem_y),
        ],
        fill=255,
    )
    draw.ellipse(
        (
            int(center_x - shoulder_width * 0.155),
            int(collar_y - shoulder_width * 0.075),
            int(center_x + shoulder_width * 0.155),
            int(collar_y + shoulder_width * 0.070),
        ),
        fill=0,
    )
    return mask.filter(ImageFilter.GaussianBlur(radius=1.2))


def controlled_base_color(garment: Image.Image):
    pixels = list(garment.convert("RGB").resize((180, 180)).getdata())
    colors = [color for color in pixels if not (color[0] > 238 and color[1] > 238 and color[2] > 238)]
    dark = [color for color in colors if sum(color) / 3 < 95]
    sample = sorted(dark if len(dark) > 30 else colors, key=sum)
    if not sample:
        return (24, 24, 27)
    mid = sample[len(sample) // 2]
    return tuple(max(0, min(255, int(value))) for value in mid)


def controlled_garment_bbox(image: Image.Image):
    img = image.convert("RGB")
    width, height = img.size
    x0, y0, x1, y1 = width, height, -1, -1
    pixels = img.load()
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            r, g, b = pixels[x, y]
            if r > 238 and g > 238 and b > 238:
                continue
            if max(r, g, b) - min(r, g, b) < 4 and max(r, g, b) > 230:
                continue
            x0, y0 = min(x0, x), min(y0, y)
            x1, y1 = max(x1, x), max(y1, y)
    if x1 < x0:
        return (0, 0, width, height)
    return (max(0, x0 - 4), max(0, y0 - 4), min(width, x1 + 5), min(height, y1 + 5))


def controlled_print(garment: Image.Image, base_color):
    crop = garment.convert("RGB").crop(controlled_garment_bbox(garment))
    width, height = crop.size
    pixels = crop.load()
    br, bg, bb = base_color
    x0, y0, x1, y1 = width, height, -1, -1
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r > 242 and g > 242 and b > 242:
                continue
            if abs(r - br) + abs(g - bg) + abs(b - bb) < 70:
                continue
            x0, y0 = min(x0, x), min(y0, y)
            x1, y1 = max(x1, x), max(y1, y)

    if x1 < x0:
        x0, x1 = int(width * 0.24), int(width * 0.76)
        y0, y1 = int(height * 0.22), int(height * 0.72)
    else:
        pad_x, pad_y = int((x1 - x0 + 1) * 0.08), int((y1 - y0 + 1) * 0.08)
        x0, x1 = max(0, x0 - pad_x), min(width - 1, x1 + pad_x)
        y0, y1 = max(0, y0 - pad_y), min(height - 1, y1 + pad_y)

    output = crop.crop((x0, y0, x1 + 1, y1 + 1)).convert("RGBA")
    alpha = Image.new("L", output.size, 0)
    alpha_pixels = alpha.load()
    output_pixels = output.load()
    for y in range(output.height):
        for x in range(output.width):
            r, g, b, _ = output_pixels[x, y]
            if not (r > 245 and g > 245 and b > 245):
                alpha_pixels[x, y] = 255
    output.putalpha(alpha.filter(ImageFilter.GaussianBlur(radius=0.4)))
    return output


def render_controlled_tryon(person: Image.Image, garment: Image.Image, pose_data: str) -> Image.Image:
    pose = read_pose(pose_data, person.size)
    if not pose:
        return person.convert("RGB")

    base = person.convert("RGBA")
    width, height = base.size
    shoulder_width = pose["shoulder_width"]
    center_x = int(pose["shoulder_center"][0] * 0.66 + pose["hip_center"][0] * 0.34)
    collar_y = int(pose["collar_y"])
    shoulder_y = int(pose["shoulder_center"][1] - shoulder_width * 0.075)
    base_color = controlled_base_color(garment)

    shirt_mask = controlled_upper_mask(person, pose_data)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(Image.new("RGBA", base.size, (*base_color, 245)), (0, 0), shirt_mask)

    shade = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shade_pixels = shade.load()
    for x in range(width):
        distance = abs(x - center_x) / max(1, shoulder_width)
        alpha = int(max(0, min(42, (distance - 0.25) * 58)))
        if not alpha:
            continue
        for y in range(max(0, shoulder_y), min(height, int(shoulder_y + shoulder_width * 2.05)), 2):
            shade_pixels[x, y] = shade_pixels[x, y + 1 if y + 1 < height else y] = (0, 0, 0, alpha)
    layer = Image.alpha_composite(layer, Image.composite(shade, Image.new("RGBA", base.size, (0, 0, 0, 0)), shirt_mask))

    print_layer = controlled_print(garment, base_color)
    target_width = int(shoulder_width * 0.62)
    target_height = int(target_width * print_layer.height / max(1, print_layer.width))
    max_height = int(shoulder_width * 0.88)
    if target_height > max_height:
        target_height = max_height
        target_width = int(max_height * print_layer.width / max(1, print_layer.height))
    print_layer = print_layer.resize((max(24, target_width), max(24, target_height)), Image.LANCZOS)
    layer.alpha_composite(print_layer, (int(center_x - target_width / 2), int(collar_y + shoulder_width * 0.38)))

    draw = ImageDraw.Draw(layer)
    collar_width, collar_height = shoulder_width * 0.34, shoulder_width * 0.15
    trim = tuple(max(0, min(255, int(channel * 0.72))) for channel in base_color)
    draw.arc(
        (
            center_x - collar_width / 2,
            collar_y - collar_height * 0.35,
            center_x + collar_width / 2,
            collar_y + collar_height * 0.70,
        ),
        start=180,
        end=360,
        fill=(*trim, 235),
        width=max(2, int(shoulder_width * 0.026)),
    )
    return Image.alpha_composite(base, layer).convert("RGB")


def infer_upper_style(garment: Image.Image, clothing_type: str) -> str:
    kind = (clothing_type or "t-shirt").strip().lower()
    if kind in {"jersey", "sports-shirt", "sports shirt", "football-shirt", "football shirt"}:
        return "jersey"
    return "t-shirt"


def make_skin_protect_mask(
    person: Image.Image,
    sleeve_bottom_y: float,
    cx: float,
    torso_guard: float,
) -> Image.Image:
    person = resize_and_crop(person.convert("RGB"), (WIDTH, HEIGHT))
    rgb = np.array(person)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    skin = (
        (cr > 132) & (cr < 178) &
        (cb > 82) & (cb < 135) &
        (y > 45) &
        (s > 18) & (s < 170) &
        (v > 55) &
        ((h < 26) | (h > 165))
    )

    yy, xx = np.indices((HEIGHT, WIDTH))
    below_sleeve = yy > max(sleeve_bottom_y - 12, HEIGHT * 0.36)
    outside_torso = (xx < cx - torso_guard) | (xx > cx + torso_guard)
    protect = (skin & below_sleeve & outside_torso).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    protect = cv2.morphologyEx(protect, cv2.MORPH_OPEN, kernel)
    protect = cv2.dilate(protect, kernel, iterations=1)
    return Image.fromarray(protect, mode="L").filter(ImageFilter.GaussianBlur(4))


def garment_fit_factors(garment: Image.Image):
    rgba = garment.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3]
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    lum = rgb.mean(axis=2)
    sat = hsv[:, :, 1]
    foreground = (alpha > 16) & ~((lum > 238) & (sat < 28))
    ys, xs = np.where(foreground)
    if len(xs) < 100:
        return {"width": 1.0, "sleeve_width": 1.0, "sleeve_length": 1.0, "body_length": 1.0}

    bbox_w = float(xs.max() - xs.min() + 1)
    bbox_h = float(ys.max() - ys.min() + 1)
    aspect = bbox_w / max(1.0, bbox_h)
    # Product photos of boxy/oversized shirts have a larger width/height ratio.
    # Keep the response conservative so the model cannot create shoulder blobs.
    width_factor = float(np.clip(0.94 + (aspect - 0.70) * 0.34, 0.90, 1.12))
    sleeve_width_factor = float(np.clip(0.96 + (aspect - 0.70) * 0.28, 0.92, 1.10))
    sleeve_length_factor = float(np.clip(0.98 + (aspect - 0.70) * 0.18, 0.94, 1.08))
    body_length_factor = float(np.clip(1.00 + ((bbox_h / max(1.0, rgba.height)) - 0.62) * 0.18, 0.94, 1.08))
    return {
        "width": width_factor,
        "sleeve_width": sleeve_width_factor,
        "sleeve_length": sleeve_length_factor,
        "body_length": body_length_factor,
    }


def make_upper_mask(
    person: Image.Image,
    clothing_type: str = "t-shirt",
    fit: dict | None = None,
) -> Image.Image:
    fit = fit or {}
    width_fit = float(fit.get("width", 1.0))
    sleeve_width_fit = float(fit.get("sleeve_width", 1.0))
    sleeve_length_fit = float(fit.get("sleeve_length", 1.0))
    body_length_fit = float(fit.get("body_length", 1.0))
    person = resize_and_crop(person.convert("RGB"), (WIDTH, HEIGHT))
    face = detect_face(person)
    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(mask)
    kind = (clothing_type or "t-shirt").strip().lower()
    is_jersey = kind in {"jersey", "sports-shirt", "sports shirt", "football-shirt", "football shirt"}
    is_tshirt = not is_jersey

    if face is not None:
        x, y, w, h = face
        cx = x + w / 2
        neck_top = y + h * (1.06 if is_tshirt else 0.98)
        shoulder_y = y + h * (1.34 if is_tshirt else 1.36)
        sleeve_y = y + h * (1.70 if is_tshirt else 1.82)
        sleeve_bottom_y = y + h * (2.42 if is_tshirt else 2.62) * sleeve_length_fit
        bottom = y + h * (4.75 if is_tshirt else 4.55) * body_length_fit
        neck_width = w * (0.32 if is_tshirt else 0.42)
        shoulder = min(w * (1.44 if is_tshirt else 1.22) * width_fit, WIDTH * (0.30 if is_tshirt else 0.24) * width_fit)
        sleeve = min(w * (1.56 if is_tshirt else 1.58) * sleeve_width_fit, WIDTH * (0.31 if is_tshirt else 0.31) * sleeve_width_fit)
        waist = min(w * (1.08 if is_tshirt else 1.00) * width_fit, WIDTH * (0.26 if is_tshirt else 0.24) * width_fit)
    else:
        cx = WIDTH / 2
        neck_top = HEIGHT * (0.25 if is_tshirt else 0.24)
        shoulder_y = HEIGHT * (0.30 if is_tshirt else 0.31)
        sleeve_y = HEIGHT * (0.38 if is_tshirt else 0.38)
        sleeve_bottom_y = HEIGHT * (0.47 if is_tshirt else 0.49) * sleeve_length_fit
        bottom = HEIGHT * (0.70 if is_tshirt else 0.64) * body_length_fit
        neck_width = WIDTH * (0.065 if is_tshirt else 0.075)
        shoulder = WIDTH * (0.28 if is_tshirt else 0.23) * width_fit
        sleeve = WIDTH * (0.31 if is_tshirt else 0.31) * sleeve_width_fit
        waist = WIDTH * (0.24 if is_tshirt else 0.20) * width_fit

    neck_top = max(0, min(HEIGHT - 1, neck_top))
    shoulder_y = max(neck_top + 30, min(HEIGHT - 1, shoulder_y))
    sleeve_y = max(shoulder_y + 25, min(HEIGHT - 1, sleeve_y))
    sleeve_bottom_y = max(sleeve_y + 55, min(HEIGHT - 1, sleeve_bottom_y))
    bottom = max(sleeve_y + 140, min(HEIGHT - 1, bottom))
    torso_at_sleeve = waist + (sleeve - waist) * 0.62
    draw.polygon(
        [
            (cx - neck_width, neck_top),
            (cx + neck_width, neck_top),
            (cx + shoulder, shoulder_y),
            (cx + sleeve, sleeve_y),
            (cx + waist, bottom),
            (cx - waist, bottom),
            (cx - sleeve, sleeve_y),
            (cx - shoulder, shoulder_y),
        ],
        fill=255,
    )
    draw.polygon(
        [
            (cx - shoulder * 0.92, shoulder_y),
            (cx - sleeve, sleeve_y),
            (cx - sleeve * 0.96, sleeve_bottom_y),
            (cx - torso_at_sleeve, sleeve_bottom_y),
            (cx - shoulder * 0.72, sleeve_y),
        ],
        fill=255,
    )
    draw.polygon(
        [
            (cx + shoulder * 0.92, shoulder_y),
            (cx + sleeve, sleeve_y),
            (cx + sleeve * 0.96, sleeve_bottom_y),
            (cx + torso_at_sleeve, sleeve_bottom_y),
            (cx + shoulder * 0.72, sleeve_y),
        ],
        fill=255,
    )

    if face is not None:
        if is_tshirt:
            neck_box = [cx - w * 0.30, y + h * 0.86, cx + w * 0.30, y + h * 1.22]
        else:
            neck_box = [cx - w * 0.34, y + h * 0.90, cx + w * 0.34, y + h * 1.35]
    else:
        if is_tshirt:
            neck_box = [WIDTH * 0.455, HEIGHT * 0.205, WIDTH * 0.545, HEIGHT * 0.290]
        else:
            neck_box = [WIDTH * 0.445, HEIGHT * 0.205, WIDTH * 0.555, HEIGHT * 0.300]
    draw.ellipse(neck_box, fill=0)

    skin_protect = make_skin_protect_mask(person, sleeve_bottom_y, cx, waist * 0.86)
    mask_arr = np.array(mask, dtype=np.uint8)
    skin_arr = np.array(skin_protect, dtype=np.uint8)
    mask_arr[skin_arr > 64] = 0
    return Image.fromarray(mask_arr, mode="L").filter(ImageFilter.GaussianBlur(5))


def garment_is_dark(garment: Image.Image) -> bool:
    rgba = garment.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3]
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    lum = rgb.mean(axis=2)
    sat = hsv[:, :, 1]
    foreground = (alpha > 16) & ~((lum > 238) & (sat < 28))
    fabric = foreground & (sat < 95) & (lum < 155)
    if fabric.sum() < 300:
        fabric = foreground & (lum < 155)
    if fabric.sum() < 300:
        return False
    return float(np.median(lum[fabric])) < 85


def clamp_mask_to_garment_fit(mask: Image.Image, person: Image.Image, garment: Image.Image, clothing_type: str) -> Image.Image:
    style = infer_upper_style(garment, clothing_type)
    fit_mask = make_upper_mask(person, style, garment_fit_factors(garment)).resize((WIDTH, HEIGHT)).convert("L")
    mask_arr = np.array(mask.resize((WIDTH, HEIGHT)).convert("L"), dtype=np.uint8)
    fit_arr = np.array(fit_mask, dtype=np.uint8)
    clipped = np.minimum(mask_arr, fit_arr)
    fit_area = max(1, int((fit_arr > 48).sum()))
    clipped_area = int((clipped > 48).sum())

    # Dark T-shirts need a full clean garment region. Auto masks can be too
    # conservative on striped/light source shirts, which leaves the old shirt
    # visible and creates dark fragments around the arms.
    if garment_is_dark(garment) or clipped_area < fit_area * 0.58:
        clipped = fit_arr.copy()

    # Keep the final mask compact. This removes shoulder/arm blobs while preserving
    # normal sleeve length inside the garment envelope.
    clipped[clipped < 32] = 0
    kernel = np.ones((5, 5), np.uint8)
    clipped = cv2.morphologyEx(clipped, cv2.MORPH_OPEN, kernel)
    clipped = cv2.dilate(clipped, kernel, iterations=1)
    clipped = np.minimum(clipped, fit_arr)
    return Image.fromarray(clipped, mode="L").filter(ImageFilter.GaussianBlur(4))


def estimate_fabric_color(garment: Image.Image) -> np.ndarray:
    arr = np.array(garment.convert("RGB")).astype(np.float32)
    hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV)
    lum = arr.mean(axis=2)
    sat = hsv[:, :, 1]
    white_bg = (lum > 235) & (sat < 35)
    valid = ~white_bg
    fabric = valid & (lum < 85) & (sat < 105)
    if fabric.sum() < 200:
        fabric = valid & (lum < 115)
    if fabric.sum() < 200:
        fabric = valid
    if fabric.sum() == 0:
        return np.array([18, 18, 18], dtype=np.float32)
    colors = arr[fabric]
    color_lum = colors.mean(axis=1)
    dark_cutoff = np.percentile(color_lum, 38)
    darkest = colors[color_lum <= dark_cutoff]
    if len(darkest) >= 100:
        colors = darkest
    return np.percentile(colors, 35, axis=0).astype(np.float32)


def preserve_fabric_color(result: Image.Image, garment: Image.Image, mask: Image.Image) -> Image.Image:
    target = estimate_fabric_color(garment)
    target_lum = float(target.mean())
    if target_lum > 120:
        return result

    arr = np.array(result.convert("RGB")).astype(np.float32)
    alpha = np.array(mask.resize(result.size).convert("L")).astype(np.float32) / 255.0
    lum = arr.mean(axis=2)
    fabric = (alpha > 0.56) & (lum < 215)
    if fabric.sum() < 100:
        return result

    current_lum = max(1.0, float(np.median(lum[fabric])))
    factor = np.clip((target_lum + 18.0) / current_lum, 0.18, 0.62)
    darkened = arr * factor
    color_layer = target.reshape(1, 1, 3)
    corrected = darkened * 0.78 + color_layer * 0.22
    strength = np.clip((alpha - 0.48) / 0.52, 0.0, 1.0)[..., None] * fabric[..., None].astype(np.float32) * 0.78
    out = arr * (1.0 - strength) + corrected * strength
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def enforce_dark_tshirt_finish(result: Image.Image, garment: Image.Image, mask: Image.Image) -> Image.Image:
    if not garment_is_dark(garment):
        return result

    arr = np.array(result.convert("RGB"), dtype=np.float32)
    alpha = np.array(mask.resize(result.size).convert("L"), dtype=np.float32) / 255.0
    shirt_pixels = alpha > 0.58
    if shirt_pixels.sum() < 500:
        return result

    ys, xs = np.where(shirt_pixels)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    w = max(1.0, x1 - x0 + 1.0)
    h = max(1.0, y1 - y0 + 1.0)
    yy, xx = np.indices(alpha.shape)

    # Keep the central product graphic readable, but recolor shoulders, sleeves,
    # side seams, and empty fabric area to the garment's real fabric color.
    logo_cx = x0 + w * 0.50
    logo_cy = y0 + h * 0.50
    logo_rx = w * 0.31
    logo_ry = h * 0.31
    print_safe = (((xx - logo_cx) / logo_rx) ** 2 + ((yy - logo_cy) / logo_ry) ** 2) <= 1.0

    target = estimate_fabric_color(garment)
    if float(target.mean()) > 42:
        target = target * (34.0 / max(1.0, float(target.mean())))
    target = np.clip(target, 8, 42).reshape(1, 1, 3)

    lum = arr.mean(axis=2)
    shade = np.clip(lum / 72.0, 0.52, 1.10)[..., None]
    fabric_color = target * shade
    recolor_area = shirt_pixels & ~print_safe

    # Bright speckles outside the logo are usually leaked source-shirt/background.
    bright_leak = shirt_pixels & (lum > 105) & ~print_safe
    recolor_strength = np.where(bright_leak, 0.96, 0.82).astype(np.float32)
    strength = recolor_area[..., None].astype(np.float32) * recolor_strength[..., None]
    out = arr * (1.0 - strength) + fabric_color * strength
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def restore_outside_tryon_region(result: Image.Image, person: Image.Image, mask: Image.Image) -> Image.Image:
    result_arr = np.array(result.convert("RGB"), dtype=np.float32)
    original_arr = np.array(resize_and_crop(person.convert("RGB"), result.size), dtype=np.float32)
    alpha = np.array(mask.resize(result.size).convert("L"), dtype=np.float32) / 255.0
    keep_generated = np.clip((alpha - 0.58) / 0.28, 0.0, 1.0)[..., None]
    out = original_arr * (1.0 - keep_generated) + result_arr * keep_generated
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def normalize_category(category: str) -> str:
    value = (category or "upper").strip().lower()
    if value in {"pants", "lower", "lower_body"}:
        return "lower"
    if value in {"dress", "dresses", "overall"}:
        return "overall"
    return "upper"


@api.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "catvton-kaggle",
        "device": DEVICE,
        "masker": "automasker" if RUNTIME.get("masker") is not None else "fallback",
        "upperDefault": "ai-generated-garment-fit",
        "controlledAvailable": True,
    }


@api.post("/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("upper"),
    clothing_type: str = Form("t-shirt"),
    garment_view: str = Form("front"),
    pose_landmarks_data: str = Form(""),
    controlled_only: bool = Form(False),
    denoise_steps: int = Form(24),
    guidance_scale: float = Form(3.0),
    seed: int = Form(42),
):
    del garment_view
    person = Image.open(io.BytesIO(await person_image.read())).convert("RGB")
    garment = Image.open(io.BytesIO(await garment_image.read())).convert("RGB")
    cat = normalize_category(category)
    person_resized = resize_and_crop(person, (WIDTH, HEIGHT))
    garment_resized = resize_and_padding(garment, (WIDTH, HEIGHT))
    upper_style = infer_upper_style(garment, clothing_type)
    upper_fit = garment_fit_factors(garment)

    if cat == "upper" and controlled_only and read_pose(pose_landmarks_data, person.size):
        result = render_controlled_tryon(person, garment, pose_landmarks_data)
        output = io.BytesIO()
        result.save(output, format="PNG")
        return {"outputImageBase64": base64.b64encode(output.getvalue()).decode("utf-8")}

    if cat == "upper":
        if RUNTIME.get("masker") is not None:
            try:
                mask = RUNTIME["masker"](person_resized, mask_type="upper")["mask"].convert("L")
                mask = mask.resize((WIDTH, HEIGHT))
                mask = clamp_mask_to_garment_fit(mask, person, garment, clothing_type)
            except Exception as exc:
                print(f"AutoMasker inference failed, using fallback mask: {exc}")
                mask = make_upper_mask(person, upper_style, upper_fit)
        else:
            mask = make_upper_mask(person, upper_style, upper_fit)
    else:
        mask = Image.new("L", (WIDTH, HEIGHT), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle(
            [WIDTH * 0.25, HEIGHT * 0.18, WIDTH * 0.75, HEIGHT * 0.92],
            radius=40,
            fill=255,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(7))

    generator = torch.Generator(device=DEVICE).manual_seed(int(seed)) if int(seed) != -1 else None
    with torch.inference_mode():
        result = PIPELINE(
            image=person_resized,
            condition_image=garment_resized,
            mask=mask,
            num_inference_steps=int(denoise_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            width=WIDTH,
            height=HEIGHT,
        )[0]

    result = repaint_result(result, person_resized, mask)
    result = preserve_fabric_color(result, garment_resized, mask)
    result = enforce_dark_tshirt_finish(result, garment_resized, mask)
    result = restore_outside_tryon_region(result, person_resized, mask)
    output = io.BytesIO()
    result.save(output, format="PNG")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"outputImageBase64": base64.b64encode(output.getvalue()).decode("utf-8")}


def install_cloudflared(path: Path):
    if path.exists() and path.stat().st_size > 1_000_000:
        return
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    urllib.request.urlretrieve(url, path)
    path.chmod(0o755)


def run_tunnel(port: int):
    cloudflared = Path("/kaggle/working/cloudflared")
    install_cloudflared(cloudflared)
    process = subprocess.Popen(
        [str(cloudflared), "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    public_url = None
    start = time.time()
    while time.time() - start < 60:
        line = process.stdout.readline() if process.stdout else ""
        if line:
            print(line, end="", flush=True)
            match = re.search(r"https://[-a-zA-Z0-9.]+\.trycloudflare\.com", line)
            if match:
                public_url = match.group(0)
                break
        time.sleep(0.1)

    if not public_url:
        raise RuntimeError("Cloudflare tunnel URL was not created. Re-run this command.")

    print("\nHealth URL:", f"{public_url}/health")
    print("TRYON URL:", f"{public_url}/tryon")
    print("Paste the TRYON URL in the website Upload page.")
    return process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tunnel", action="store_true")
    args = parser.parse_args()

    if args.tunnel:
        run_tunnel(args.port)
    uvicorn.run(api, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
