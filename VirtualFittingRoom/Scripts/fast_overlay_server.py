import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from PIL import Image


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


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def remove_white_background(garment_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    coords = cv2.findNonZero(mask)
    if coords is None:
        full_mask = np.full(garment_bgr.shape[:2], 255, dtype=np.uint8)
        return garment_bgr, full_mask

    x, y, w, h = cv2.boundingRect(coords)
    cropped_garment = garment_bgr[y:y + h, x:x + w]
    cropped_mask = mask[y:y + h, x:x + w]
    return cropped_garment, cropped_mask


def detect_face(person_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None

    return max(faces, key=lambda face: face[2] * face[3])


def estimate_upper_body_quad(person_bgr: np.ndarray) -> np.ndarray:
    height, width = person_bgr.shape[:2]
    face = detect_face(person_bgr)

    if face is not None:
        x, y, w, h = face
        center_x = x + (w / 2.0)
        shoulders_y = y + (h * 1.95)
        waist_y = y + (h * 5.25)
        top_half_width = w * 1.2
        bottom_half_width = top_half_width * 0.78
    else:
        center_x = width * 0.5
        shoulders_y = height * 0.21
        waist_y = height * 0.66
        top_half_width = width * 0.17
        bottom_half_width = width * 0.14

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


def estimate_target_quad(person_bgr: np.ndarray, category: str) -> np.ndarray:
    if category == "lower":
        return estimate_lower_body_quad(person_bgr)
    if category == "overall":
        upper = estimate_upper_body_quad(person_bgr)
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
    return estimate_upper_body_quad(person_bgr)


def warp_garment_to_person(person_bgr: np.ndarray, garment_bgr: np.ndarray, category: str) -> np.ndarray:
    garment_bgr, garment_mask = remove_white_background(garment_bgr)
    person_h, person_w = person_bgr.shape[:2]
    garment_h, garment_w = garment_bgr.shape[:2]

    if category == "upper":
        src_points = np.array(
            [
                [garment_w * 0.16, garment_h * 0.18],
                [garment_w * 0.84, garment_h * 0.18],
                [garment_w * 0.72, garment_h * 0.98],
                [garment_w * 0.28, garment_h * 0.98],
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

    dst_points = estimate_target_quad(person_bgr, category)

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

    blurred_alpha = cv2.GaussianBlur(warped_mask, (7, 7), 0).astype(np.float32) / 255.0
    alpha = np.clip(blurred_alpha[..., None] * 0.95, 0.0, 1.0)

    composite = (warped_garment.astype(np.float32) * alpha) + (person_bgr.astype(np.float32) * (1.0 - alpha))
    return np.clip(composite, 0, 255).astype(np.uint8)


def run_tryon(person_bytes: bytes, cloth_bytes: bytes, category: str) -> bytes:
    person_image = Image.open(io.BytesIO(person_bytes)).convert("RGB")
    cloth_image = Image.open(io.BytesIO(cloth_bytes)).convert("RGB")

    person_bgr = pil_to_bgr(person_image)
    cloth_bgr = pil_to_bgr(cloth_image)
    result_bgr = warp_garment_to_person(person_bgr, cloth_bgr, category)

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
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))

            person_image = base64.b64decode(payload["personImageBase64"])
            clothing_image = base64.b64decode(payload["clothingImageBase64"])
            category = payload.get("category", "upper")

            output_bytes = run_tryon(person_image, clothing_image, category)
            return self._write_json(200, {
                "outputImageBase64": base64.b64encode(output_bytes).decode("utf-8")
            })
        except Exception as ex:
            return self._write_json(500, {"error": str(ex)})

    def log_message(self, format, *args):
        return

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
