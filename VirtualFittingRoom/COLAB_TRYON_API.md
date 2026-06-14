# Google Colab CatVTON FastAPI

Use this when Hugging Face CPU is too slow and you need the real generated try-on result instead of a pasted overlay.

## 1. Open the notebook

Upload and open this notebook in Google Colab:

`../Colab/Virtual_Try_On_API_Colab.ipynb`

In Colab:

1. Runtime > Change runtime type
2. Choose `T4 GPU`
3. Run all cells
4. When asked, upload `../Colab/colab_tryon_api.py`
5. Wait for `Model is ready`
6. Copy the public URL printed by the final cell

The final URL looks like:

`https://something.trycloudflare.com/tryon`

## 2. Use It In The Website

Open the website Upload page and paste the URL into:

`Colab FastAPI URL`

Then upload the model image and clothing image and click Generate.

Optional global config, if you do not want to paste the URL every time:

```powershell
setx VirtualTryOn__Mode "Api"
setx VirtualTryOn__ApiUrl "PASTE_COLAB_URL_HERE/tryon"
```

Close and reopen Visual Studio or the terminal after running `setx`.

## Notes

- Colab free sessions are temporary. If Colab disconnects, run the notebook again and copy the new URL.
- Keep the Colab tab open while using the website.
- First startup downloads and loads the model, so it takes a few minutes. Inference after loading is the fast part.
- If the output is still pasted-looking, you are using the old lightweight notebook. Use `Colab/Virtual_Try_On_API_Colab.ipynb`.
