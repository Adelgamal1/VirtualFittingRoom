# Colab Virtual Try-On API

Use this when the laptop cannot run the try-on AI locally.

1. Open Google Colab and choose a GPU runtime.
2. Run these cells:

```python
!git clone https://github.com/Zheng-Chong/CatVTON /content/CatVTON
!pip install -q fastapi uvicorn python-multipart huggingface_hub diffusers transformers accelerate safetensors opencv-python pillow scipy tqdm
```

3. Upload `Colab/colab_tryon_api.py` to Colab, then run:

```python
!python /content/colab_tryon_api.py --tunnel --width 576 --height 768 --steps 20
```

4. Copy the printed URL ending with `/tryon`.
5. Wait until Colab prints `Model is ready` before sending images from the website.
6. Put it in `VirtualFittingRoom/appsettings.json`:

```json
"Mode": "Api",
"ApiUrl": "https://YOUR-COLAB-TUNNEL.trycloudflare.com/tryon"
```

The ASP.NET site already sends:
- `person_image`
- `garment_image`
- `category`

The Colab API returns a PNG image directly.
