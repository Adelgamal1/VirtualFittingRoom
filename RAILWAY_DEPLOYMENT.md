# Railway Deployment

## What this deploys

This deploys the ASP.NET Core MVC web app in `VirtualFittingRoom/`.

Do not run the heavy local AI model inside the Railway web service. Keep try-on inference on Hugging Face, Replicate, or another public API and configure the MVC app with environment variables.

## Files added for Railway

- `Dockerfile`: builds and runs the .NET 6 MVC app.
- `.dockerignore`: keeps notebooks, local environments, and model weights out of the Docker build.
- `railway.json`: tells Railway to use the Dockerfile and checks `/healthz`.
- `VirtualFittingRoom/runtimeconfig.template.json`: enables .NET 6 `System.Drawing` support on Linux.

## Required Railway variables

Set these in the Railway service Variables tab:

```text
ASPNETCORE_ENVIRONMENT=Production
ConnectionStrings__DefaultConnection=Server=tcp:YOUR_SQL_SERVER.database.windows.net,1433;Initial Catalog=VFR_DB;Persist Security Info=False;User ID=YOUR_SQL_USER;Password=YOUR_SQL_PASSWORD;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;
```

The app currently uses SQL Server via Entity Framework. Railway provides managed PostgreSQL/MySQL/Redis/MongoDB, not SQL Server, so use an external SQL Server such as Azure SQL unless you choose to refactor the app to another database provider.

## AI variables

For Hugging Face Spaces:

```text
VirtualTryOn__Mode=HuggingFace
VirtualTryOn__HuggingFaceSpaceUrl=https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space
VirtualTryOn__HuggingFaceApiName=tryon
VirtualTryOn__HuggingFaceToken=
```

For Replicate:

```text
VirtualTryOn__Mode=Replicate
REPLICATE_API_TOKEN=r8_YOUR_REAL_TOKEN
VirtualTryOn__ReplicateVersion=cf5cb07a25e726fe2fac166a8c5ab52ddccd48657741670fb09d9954d4d8446f
```

Use only one AI mode at a time.

## Deploy from GitHub

1. Commit and push the project to GitHub.
2. Open Railway and create a new project.
3. Choose `Deploy from GitHub repo`.
4. Select this repository.
5. After deployment succeeds, open the service Settings, go to Networking, and generate a public domain.

Railway detects the root `Dockerfile`. The app binds to Railway's `PORT` variable automatically.

## Deploy from Railway CLI

From the repository root:

```powershell
railway login
railway init
railway up
```

If Railway asks for a service root, use the repository root because the `Dockerfile` is there.

## Common failure checks

- If you see `Application failed to respond`, confirm the service is using this `Dockerfile` and that `/healthz` is passing.
- If login, registration, or history pages fail, check `ConnectionStrings__DefaultConnection`.
- If try-on fails, check `VirtualTryOn__Mode` and the selected AI provider token/URL.
- Do not commit local model directories or checkpoint files unless you intentionally use Git LFS and a separate inference service.
