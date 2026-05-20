# IDM-VTON Colab API

Use this notebook when you need realistic virtual try-on results instead of the lightweight overlay demo.

## What This Does

- Runs the real `yisol/IDM-VTON` model in Google Colab.
- Exposes a public `/tryon` endpoint through Cloudflare Tunnel.
- Matches the ASP.NET MVC app request format:
  - `person_image`
  - `garment_image`
  - `category`
- Returns:
  - `outputImageBase64`

## Colab Steps

1. Open Google Colab.
2. Upload:

   `idm_vton_colab_api.ipynb`

3. Runtime > Change runtime type.
4. Select `T4 GPU`.
5. Run all cells.
6. Wait for the model download and loading. The first run can take several minutes.
7. Copy the final URL ending with `/tryon`.

## Connect The MVC App

In a new PowerShell window:

```powershell
setx VirtualTryOn__Mode "Api"
setx VirtualTryOn__ApiUrl "PASTE_IDM_VTON_COLAB_URL_HERE"
setx VirtualTryOn__ApiPersonFieldName "person_image"
setx VirtualTryOn__ApiClothingFieldName "garment_image"
setx VirtualTryOn__ApiCategoryFieldName "category"
setx VirtualTryOn__ApiResponseImageField "outputImageBase64"
```

Close and reopen PowerShell or Visual Studio, then run:

```powershell
cd "C:\Users\adelm\OneDrive - Egyptian E-Learning University\Desktop\VirtualFittingRoom"
dotnet run --project .\VirtualFittingRoom\VirtualFittingRoom.csproj
```

## Important Notes

- Keep the Colab tab open while using the website.
- Free Colab can disconnect or run out of GPU memory.
- If Cloudflare gives a new URL, update `VirtualTryOn__ApiUrl`.
- If `app.Run()` fails locally, stop the old website process:

```powershell
Stop-Process -Name VirtualFittingRoom -Force
```

