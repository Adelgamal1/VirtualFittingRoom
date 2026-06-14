# Colab CatVTON FastAPI

Use this when Hugging Face CPU is too slow. This runs the real CatVTON try-on model on a Colab GPU and exposes a FastAPI `/tryon` endpoint.

1. Open Google Colab and choose a GPU runtime.
2. Upload and open:

   `Virtual_Try_On_API_Colab.ipynb`

3. Run all cells.
4. When Colab asks, upload:

   `colab_tryon_api.py`

5. Copy the printed URL ending with `/tryon`.
6. Paste it into the website Upload page under `Colab FastAPI URL`.

Manual commands, if you do not use the notebook:

```python
!if [ ! -d /content/CatVTON ]; then git clone --depth 1 https://github.com/Zheng-Chong/CatVTON /content/CatVTON; fi
!pip install -q fastapi uvicorn python-multipart huggingface_hub diffusers transformers accelerate safetensors opencv-python-headless pillow scipy tqdm
```

Upload `Colab/colab_tryon_api.py` to Colab, then run:

```python
!python /content/colab_tryon_api.py --tunnel --width 576 --height 768 --steps 12 --guidance-scale 2.0
```

The ASP.NET site sends:
- `person_image`
- `garment_image`
- `category`

The Colab API returns a PNG image directly. Keep the Colab tab open while using the website.
