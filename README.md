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
