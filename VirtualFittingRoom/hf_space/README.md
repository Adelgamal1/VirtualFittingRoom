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

Create the new Space here: https://huggingface.co/new-space

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

## Fast natural T-shirt mode

This Space defaults to a fast controlled renderer for upper-body T-shirts:

- `VFR_FAST_CONTROLLED_ONLY=1` skips the diffusion model for upper garments and returns a fitted T-shirt render quickly.
- `VFR_FORCE_CONTROLLED_UPPER=1` uses the controlled renderer for upper garments after diffusion if the fast path is disabled.

Keep those defaults for a fast result similar to the project screenshots. To test the raw CatVTON model instead, set both variables to `0` in the Space settings.
