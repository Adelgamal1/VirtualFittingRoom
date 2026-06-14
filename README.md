---
title: Virtual Fitting Room
emoji: 👕
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
---

# Virtual Fitting Room

This Hugging Face Space runs a CATVTON-based virtual fitting room demo.
The person input supports both image upload and webcam capture. The clothing input
uses image upload.

## Required project structure

Upload these folders/files into the Space repository root together with `app.py` and `requirements.txt`:

- `model/`
- `base_model/`
- `catvton_weights/`
- `utils.py`

If your folders currently have long extracted names like:

- `model-20260422T170440Z-3-001/model`
- `base_model-20260422T170410Z-3-002/base_model`
- `catvton_weights-20260422T170432Z-3-001/catvton_weights`

rename or copy them in the Space repo root to exactly:

- `model`
- `base_model`
- `catvton_weights`

## Notes

- Free CPU Spaces may be slow for high-quality inference.
- For better results, GPU hardware is recommended.
- If `DensePose` or `SCHP` dependencies are unavailable, the app falls back to a simpler mask.
- Webcam capture in the Space takes one still photo and then runs try-on. It is not
  frame-by-frame live video inference.

## Connect the MVC app to this Space

After the Space is running, set your ASP.NET app configuration like this:

```json
"VirtualTryOn": {
  "Mode": "HuggingFace",
  "HuggingFaceSpaceUrl": "https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space",
  "HuggingFaceApiName": "tryon",
  "HuggingFaceToken": ""
}
```

If the Space is private, create a Hugging Face access token and put it in the `HF_TOKEN`
environment variable instead of committing it to `appsettings.json`.

The Gradio API endpoint is `/call/tryon`; the MVC service already sends the inputs in
the order this Space expects.
