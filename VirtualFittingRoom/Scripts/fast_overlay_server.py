import argparse
import base64
import io
import importlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from PIL import Image

try:
    import mediapipe as mp
except Exception:
    mp = None


def resolve_mediapipe_pose():
    """Return the classic MediaPipe Pose module when this install exposes it."""
    if mp is not None:
        solutions = getattr(mp, "solutions", None)
        pose_module = getattr(solutions, "pose", None)
        if pose_module is not None:
            return pose_module

    for module_name in ("mediapipe.solutions.pose", "mediapipe.python.solutions.pose"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue

    return None


MP_POSE = resolve_mediapipe_pose()


def parse_args():
    parser = argparse.ArgumentParser(description="Fast pose-aware garment overlay server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5011)
    parser.add_argument("--project-root", default="")
    parser.add_argument("--base-model-path", default="")
    parser.add_argument("--resume-path", default="")
    parser.add_argument("--attn-version", default="mix")
    return parser.parse_args()


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def pil_to_bgr_and_alpha(image: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    rgba = np.array(image.convert("RGBA"))
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = rgba[:, :, 3]
    if int(alpha.min()) >= 250:
        return bgr, None
    return bgr, alpha


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def fit_image(image: Image.Image, max_side: int = 900) -> Image.Image:
    width, height = image.size
    largest_side = max(width, height)
    if largest_side <= max_side:
        return image

    scale = max_side / largest_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def crop_to_mask(garment_bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    coords = cv2.findNonZero(mask)
    if coords is None or cv2.countNonZero(mask) < int(mask.size * 0.02):
        return None

    x, y, w, h = cv2.boundingRect(coords)
    pad = max(3, int(max(w, h) * 0.015))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(garment_bgr.shape[1], x + w + pad)
    y1 = min(garment_bgr.shape[0], y + h + pad)
    return garment_bgr[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def remove_white_background(garment_bgr: np.ndarray, alpha_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if alpha_mask is not None:
        mask = cv2.threshold(alpha_mask, 96, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        cropped = crop_to_mask(garment_bgr, mask)
        if cropped is not None:
            return cropped

    garment_rgb = cv2.cvtColor(garment_bgr, cv2.COLOR_BGR2RGB)
    lower = np.array([0, 0, 0], dtype=np.uint8)
    upper = np.array([242, 242, 242], dtype=np.uint8)
    non_white_mask = cv2.inRange(garment_rgb, lower, upper)

    hsv = cv2.cvtColor(garment_bgr, cv2.COLOR_BGR2HSV)
    low_sat_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 40, 255]))
    mask = cv2.bitwise_and(non_white_mask, cv2.bitwise_not(low_sat_mask))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    cropped = crop_to_mask(garment_bgr, mask)
    coords = cv2.findNonZero(mask)
    if cropped is None:
        rect_margin_x = max(4, int(garment_bgr.shape[1] * 0.04))
        rect_margin_y = max(4, int(garment_bgr.shape[0] * 0.03))
        grabcut_mask = np.zeros(garment_bgr.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (
            rect_margin_x,
            rect_margin_y,
            max(1, garment_bgr.shape[1] - (rect_margin_x * 2)),
            max(1, garment_bgr.shape[0] - (rect_margin_y * 2)),
        )
        try:
            cv2.grabCut(garment_bgr, grabcut_mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
            mask = np.where(
                (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype("uint8")
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            coords = cv2.findNonZero(mask)
            cropped = crop_to_mask(garment_bgr, mask)
        except Exception:
            coords = None

    if cropped is None or coords is None:
        full_mask = np.full(garment_bgr.shape[:2], 255, dtype=np.uint8)
        return garment_bgr, full_mask

    return cropped


def detect_face(person_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None

    return max(faces, key=lambda face: face[2] * face[3])


def estimate_upper_body_quad(person_bgr: np.ndarray, clothing_type: str = "") -> np.ndarray:
    height, width = person_bgr.shape[:2]
    face = detect_face(person_bgr)
    is_tshirt = clothing_type == "t-shirt"
    is_jersey = clothing_type == "jersey"

    if face is not None:
        x, y, w, h = face
        center_x = x + (w / 2.0)
        face_bottom = y + h
        shoulders_y = max(face_bottom + (h * 0.58), height * 0.30)
        waist_y = shoulders_y + (h * (2.50 if is_tshirt else (2.35 if is_jersey else 2.85)))
        top_half_width = w * (1.50 if is_tshirt else (1.48 if is_jersey else 1.55))
        bottom_half_width = top_half_width * (0.82 if is_tshirt else (0.78 if is_jersey else 0.82))
    else:
        center_x = width * 0.5
        shoulders_y = height * (0.36 if is_tshirt else (0.34 if is_jersey else 0.34))
        waist_y = height * (0.80 if is_tshirt else (0.72 if is_jersey else 0.80))
        top_half_width = width * (0.28 if is_tshirt else (0.27 if is_jersey else 0.28))
        bottom_half_width = width * (0.23 if is_tshirt else (0.21 if is_jersey else 0.22))

    shoulders_y = max(0.0, min(float(height - 1), shoulders_y))
    waist_y = max(shoulders_y + 30.0, min(float(height - 1), waist_y))

    quad = np.array(
        [
            [center_x - top_half_width, shoulders_y],
            [center_x + top_half_width, shoulders_y],
            [center_x + bottom_half_width, waist_y],
            [center_x - bottom_half_width, waist_y],
        ],
        dtype=np.float32,
    )

    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    return quad


def normalize_clothing_type(value: str | None) -> str:
    text = (value or "").strip().lower().replace(" ", "-").replace("_", "-")
    return {
        "tee": "t-shirt",
        "tshirt": "t-shirt",
        "t-shirts": "t-shirt",
        "sports-shirt": "jersey",
        "sport-shirt": "jersey",
        "hockey-jersey": "jersey",
        "football-jersey": "jersey",
        "basketball-jersey": "jersey",
        "tanktop": "tank-top",
        "sleeveless": "tank-top",
        "sleeveless-shirt": "tank-top",
    }.get(text, text)


def estimate_pose_quad(person_bgr: np.ndarray, category: str, clothing_type: str = "") -> np.ndarray | None:
    if MP_POSE is None:
        return None

    height, width = person_bgr.shape[:2]
    rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
    with MP_POSE.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.55,
    ) as pose:
        results = pose.process(rgb)

    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark

    def point(index: int) -> np.ndarray | None:
        landmark = landmarks[index]
        if landmark.visibility < 0.45:
            return None
        return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)

    left_shoulder = point(11)
    right_shoulder = point(12)
    left_hip = point(23)
    right_hip = point(24)
    if left_shoulder is None or right_shoulder is None or left_hip is None or right_hip is None:
        return None

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0
    shoulder_width = float(np.linalg.norm(right_shoulder - left_shoulder))
    torso_height = float(np.linalg.norm(hip_center - shoulder_center))
    if shoulder_width < width * 0.08 or torso_height < height * 0.12:
        return None

    shoulder_vector = right_shoulder - left_shoulder
    shoulder_unit = shoulder_vector / max(1.0, np.linalg.norm(shoulder_vector))
    hip_width = max(float(np.linalg.norm(right_hip - left_hip)), shoulder_width * 0.72)

    torso_vector = hip_center - shoulder_center
    is_tshirt = clothing_type == "t-shirt"
    is_jersey = clothing_type == "jersey"
    top_center = shoulder_center + (torso_vector * (-0.03 if is_tshirt else (-0.02 if is_jersey else 0.07)))
    upper_bottom_center = shoulder_center + (torso_vector * (0.88 if is_tshirt else (0.76 if is_jersey else 0.98)))
    top_half = shoulder_width * (0.76 if is_tshirt else (0.78 if is_jersey else 0.86))
    bottom_half = max(
        hip_width * (0.52 if is_tshirt else (0.50 if is_jersey else 0.55)),
        shoulder_width * (0.52 if is_tshirt else (0.48 if is_jersey else 0.50)),
    )

    if category == "lower":
        knee_left = point(25)
        knee_right = point(26)
        ankle_left = point(27)
        ankle_right = point(28)
        lower_top_center = hip_center
        lower_bottom_center = (
            (ankle_left + ankle_right) / 2.0
            if ankle_left is not None and ankle_right is not None
            else ((knee_left + knee_right) / 2.0 if knee_left is not None and knee_right is not None else hip_center + np.array([0, torso_height * 1.75], dtype=np.float32))
        )
        return np.array(
            [
                lower_top_center - (shoulder_unit * hip_width * 0.52),
                lower_top_center + (shoulder_unit * hip_width * 0.52),
                lower_bottom_center + (shoulder_unit * hip_width * 0.38),
                lower_bottom_center - (shoulder_unit * hip_width * 0.38),
            ],
            dtype=np.float32,
        )

    if category == "overall":
        ankle_left = point(27)
        ankle_right = point(28)
        bottom_center = (
            (ankle_left + ankle_right) / 2.0
            if ankle_left is not None and ankle_right is not None
            else hip_center + ((hip_center - shoulder_center) * 1.45)
        )
        return np.array(
            [
                top_center - (shoulder_unit * top_half),
                top_center + (shoulder_unit * top_half),
                bottom_center + (shoulder_unit * max(bottom_half, shoulder_width * 0.44)),
                bottom_center - (shoulder_unit * max(bottom_half, shoulder_width * 0.44)),
            ],
            dtype=np.float32,
        )

    return np.array(
        [
            top_center - (shoulder_unit * top_half),
            top_center + (shoulder_unit * top_half),
            upper_bottom_center + (shoulder_unit * bottom_half),
            upper_bottom_center - (shoulder_unit * bottom_half),
        ],
        dtype=np.float32,
    )


def estimate_lower_body_quad(person_bgr: np.ndarray) -> np.ndarray:
    height, width = person_bgr.shape[:2]
    center_x = width * 0.5
    hips_y = height * 0.52
    ankle_y = height * 0.96
    top_half_width = width * 0.20
    bottom_half_width = width * 0.16

    quad = np.array(
        [
            [center_x - top_half_width, hips_y],
            [center_x + top_half_width, hips_y],
            [center_x + bottom_half_width, ankle_y],
            [center_x - bottom_half_width, ankle_y],
        ],
        dtype=np.float32,
    )
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    return quad


def estimate_target_quad(person_bgr: np.ndarray, category: str, clothing_type: str = "") -> np.ndarray:
    pose_quad = estimate_pose_quad(person_bgr, category, clothing_type)
    if pose_quad is not None:
        height, width = person_bgr.shape[:2]
        pose_quad[:, 0] = np.clip(pose_quad[:, 0], 0, width - 1)
        pose_quad[:, 1] = np.clip(pose_quad[:, 1], 0, height - 1)
        return pose_quad

    if category == "lower":
        return estimate_lower_body_quad(person_bgr)
    if category == "overall":
        upper = estimate_upper_body_quad(person_bgr, clothing_type)
        lower = estimate_lower_body_quad(person_bgr)
        return np.array(
            [
                upper[0],
                upper[1],
                lower[2],
                lower[3],
            ],
            dtype=np.float32,
        )
    return estimate_upper_body_quad(person_bgr, clothing_type)


def apply_body_lighting(person_bgr: np.ndarray, warped_garment: np.ndarray, solid_mask: np.ndarray) -> np.ndarray:
    person_gray = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    garment_gray = cv2.cvtColor(warped_garment, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    soft_body_light = cv2.GaussianBlur(person_gray, (0, 0), 18)
    mask_pixels = soft_body_light[solid_mask > 0]
    if mask_pixels.size == 0:
        return warped_garment

    body_mid = float(np.median(mask_pixels))
    gain = np.clip((soft_body_light / max(body_mid, 0.08)) ** 0.72, 0.72, 1.22)

    garment_edges = cv2.Laplacian(garment_gray, cv2.CV_32F)
    print_protection = np.clip(np.abs(garment_edges) * 2.2, 0.0, 1.0)
    print_protection = cv2.GaussianBlur(print_protection, (5, 5), 0)[..., None]

    shaded = warped_garment.astype(np.float32) * gain[..., None]
    shaded = (shaded * (1.0 - (print_protection * 0.26))) + (warped_garment.astype(np.float32) * (print_protection * 0.26))
    return np.clip(shaded, 0, 255).astype(np.uint8)


def build_worn_composite(person_bgr: np.ndarray, warped_garment: np.ndarray, warped_mask: np.ndarray) -> np.ndarray:
    solid_mask = cv2.threshold(warped_mask, 36, 255, cv2.THRESH_BINARY)[1]
    cleanup_kernel = np.ones((3, 3), np.uint8)
    solid_mask = cv2.morphologyEx(solid_mask, cv2.MORPH_CLOSE, cleanup_kernel, iterations=1)
    core_mask = cv2.erode(solid_mask, cleanup_kernel, iterations=1)

    edge_mask = cv2.subtract(cv2.dilate(core_mask, cleanup_kernel, iterations=2), core_mask)
    edge_alpha = cv2.GaussianBlur(edge_mask, (11, 11), 0).astype(np.float32) / 255.0

    alpha = cv2.GaussianBlur(core_mask, (9, 9), 0).astype(np.float32) / 255.0
    alpha = np.clip((alpha * 0.97) - (edge_alpha * 0.12), 0.0, 0.97)[..., None]

    shaded_garment = apply_body_lighting(person_bgr, warped_garment, core_mask)

    person_float = person_bgr.astype(np.float32)
    contact_shadow = cv2.GaussianBlur(
        cv2.dilate(core_mask, np.ones((13, 13), np.uint8), iterations=1),
        (25, 25),
        0,
    ).astype(np.float32) / 255.0
    underlay = person_float * (1.0 - (contact_shadow[..., None] * 0.10))

    composite = (shaded_garment.astype(np.float32) * alpha) + (underlay * (1.0 - alpha))

    seam_mask = cv2.GaussianBlur(edge_mask, (15, 15), 0).astype(np.float32) / 255.0
    seam_mask = np.clip(seam_mask[..., None] * 0.18, 0.0, 0.18)
    composite = (composite * (1.0 - seam_mask)) + (person_float * seam_mask)

    return np.clip(composite, 0, 255).astype(np.uint8)


def build_fit_silhouette(size: tuple[int, int], dst_points: np.ndarray, category: str, clothing_type: str) -> np.ndarray:
    width, height = size
    silhouette = np.zeros((height, width), dtype=np.uint8)
    if category == "upper" and clothing_type == "t-shirt":
        top_left, top_right, hem_right, hem_left = dst_points
        center_x = float(np.mean(dst_points[:, 0]))
        top_y = float((top_left[1] + top_right[1]) / 2.0)
        hem_y = float((hem_left[1] + hem_right[1]) / 2.0)
        garment_height = max(1.0, hem_y - top_y)
        shoulder_half = max(1.0, abs(top_right[0] - top_left[0]) / 2.0)
        hem_half = max(1.0, abs(hem_right[0] - hem_left[0]) / 2.0)

        points = np.array(
            [
                [center_x - shoulder_half, top_y + garment_height * 0.20],
                [center_x - shoulder_half * 0.62, top_y],
                [center_x + shoulder_half * 0.62, top_y],
                [center_x + shoulder_half, top_y + garment_height * 0.20],
                [center_x + hem_half, hem_y],
                [center_x - hem_half, hem_y],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(silhouette, [points], 255)

        neck_width = max(16, int(shoulder_half * 0.46))
        neck_height = max(8, int(garment_height * 0.11))
        neck_center = (int(center_x), int(top_y + neck_height * 0.65))
        cv2.ellipse(silhouette, neck_center, (neck_width // 2, neck_height), 0, 0, 360, 0, -1)

        kernel_size = max(5, int(width * 0.010))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, kernel, iterations=1)
        silhouette = cv2.GaussianBlur(silhouette, (kernel_size, kernel_size), 0)
        return silhouette

    points = dst_points.astype(np.int32)
    cv2.fillConvexPoly(silhouette, points, 255)
    return silhouette


def warp_garment_to_person(
    person_bgr: np.ndarray,
    garment_bgr: np.ndarray,
    category: str,
    alpha_mask: np.ndarray | None = None,
    clothing_type: str = "",
) -> np.ndarray:
    garment_bgr, garment_mask = remove_white_background(garment_bgr, alpha_mask)
    person_h, person_w = person_bgr.shape[:2]
    garment_h, garment_w = garment_bgr.shape[:2]

    if category == "upper" and clothing_type == "t-shirt":
        src_points = np.array(
            [
                [garment_w * 0.06, garment_h * 0.08],
                [garment_w * 0.94, garment_h * 0.08],
                [garment_w * 0.66, garment_h * 0.94],
                [garment_w * 0.34, garment_h * 0.94],
            ],
            dtype=np.float32,
        )
    elif category == "upper":
        src_points = np.array(
            [
                [garment_w * 0.06, garment_h * 0.20],
                [garment_w * 0.94, garment_h * 0.20],
                [garment_w * 0.66, garment_h * 0.94],
                [garment_w * 0.34, garment_h * 0.94],
            ],
            dtype=np.float32,
        )
    elif category == "lower":
        src_points = np.array(
            [
                [garment_w * 0.24, garment_h * 0.06],
                [garment_w * 0.76, garment_h * 0.06],
                [garment_w * 0.66, garment_h * 0.98],
                [garment_w * 0.34, garment_h * 0.98],
            ],
            dtype=np.float32,
        )
    else:
        src_points = np.array(
            [
                [garment_w * 0.18, garment_h * 0.12],
                [garment_w * 0.82, garment_h * 0.12],
                [garment_w * 0.70, garment_h * 0.98],
                [garment_w * 0.30, garment_h * 0.98],
            ],
            dtype=np.float32,
        )

    dst_points = estimate_target_quad(person_bgr, category, clothing_type)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_garment = cv2.warpPerspective(
        garment_bgr,
        matrix,
        (person_w, person_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpPerspective(
        garment_mask,
        matrix,
        (person_w, person_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if alpha_mask is None:
        fit_silhouette = build_fit_silhouette((person_w, person_h), dst_points, category, clothing_type)
        warped_mask = cv2.min(warped_mask, fit_silhouette)

    return build_worn_composite(person_bgr, warped_garment, warped_mask)


def run_tryon(person_bytes: bytes, cloth_bytes: bytes, category: str, clothing_type: str = "") -> bytes:
    person_image = fit_image(Image.open(io.BytesIO(person_bytes)).convert("RGB"))
    cloth_image = fit_image(Image.open(io.BytesIO(cloth_bytes)).convert("RGBA"), max_side=700)

    person_bgr = pil_to_bgr(person_image)
    cloth_bgr, cloth_alpha = pil_to_bgr_and_alpha(cloth_image)
    result_bgr = warp_garment_to_person(person_bgr, cloth_bgr, category, cloth_alpha, normalize_clothing_type(clothing_type))

    output = io.BytesIO()
    bgr_to_pil(result_bgr).save(output, format="PNG")
    return output.getvalue()


class TryOnHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            return self._write_json(200, {"status": "ok", "engine": "fast-overlay"})
        return self._write_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/tryon":
            return self._write_json(404, {"error": "Not found"})

        try:
            raw_body = self._read_request_body()
            if not raw_body.strip():
                return self._write_json(400, {"error": "Empty AI request body. The web app must send JSON with model and clothing images."})

            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError as ex:
                return self._write_json(400, {"error": f"Invalid AI request JSON: {ex}"})

            person_image = base64.b64decode(payload["personImageBase64"])
            clothing_image = base64.b64decode(payload["clothingImageBase64"])
            category = payload.get("category", "upper")
            clothing_type = payload.get("clothingType", "")

            output_bytes = run_tryon(person_image, clothing_image, category, clothing_type)
            return self._write_json(200, {
                "outputImageBase64": base64.b64encode(output_bytes).decode("utf-8")
            })
        except Exception as ex:
            return self._write_json(500, {"error": str(ex)})

    def log_message(self, format, *args):
        return

    def _read_request_body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            chunks = []
            while True:
                line = self.rfile.readline().strip()
                if not line:
                    continue

                chunk_size = int(line.split(b";", 1)[0], 16)
                if chunk_size == 0:
                    self.rfile.readline()
                    break

                chunks.append(self.rfile.read(chunk_size))
                self.rfile.readline()

            return b"".join(chunks)

        content_length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(content_length)

    def _write_json(self, status_code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), TryOnHandler)
    print(f"Fast overlay AI server listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
