# Google Colab Try-On API

Use this when Replicate requires credit and the local backend is unreliable.

## 1. Open the notebook

Upload and open this notebook in Google Colab:

`colab_virtual_tryon_api.ipynb`

In Colab:

1. Runtime > Change runtime type
2. Choose `T4 GPU` if available, otherwise CPU is fine for the lightweight demo
3. Run all cells
4. Copy the public URL printed by the final cell

The final URL looks like:

`https://something.trycloudflare.com/tryon`

## 2. Point the MVC app to Colab

In a new PowerShell window:

```powershell
setx VirtualTryOn__Mode "Api"
setx VirtualTryOn__ApiUrl "PASTE_COLAB_URL_HERE/tryon"
setx VirtualTryOn__ApiResponseImageField "outputImageBase64"
```

Close and reopen PowerShell or Visual Studio after running `setx`.

Then start the website:

```powershell
cd "C:\Users\adelm\OneDrive - Egyptian E-Learning University\Desktop\VirtualFittingRoom"
dotnet run --project .\VirtualFittingRoom\VirtualFittingRoom.csproj
```

Open:

`http://localhost:5001`

## Notes

- Colab free sessions are temporary. If Colab disconnects, run the notebook again and copy the new URL.
- The included notebook runs a lightweight overlay backend, not the paid Replicate generative model.
- Keep the Colab tab open while using the website.
