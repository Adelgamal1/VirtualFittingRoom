# Azure Deployment Steps

## Goal

Deploy the ASP.NET MVC website to Azure App Service, move the database to Azure SQL, and run virtual try-on through Replicate.

## Current architecture

- Website: ASP.NET MVC
- Database: currently local SQL/LocalDB
- AI: Replicate prediction API

## Important note

Do not run the heavy AI model inside Azure Web App.

Use this split:

1. Web App on Azure App Service
2. Database on Azure SQL
3. AI as Replicate hosted inference

## Phase 1 - Deploy the website first

### 1. Create a Resource Group

In Azure Portal:

1. Open `Resource groups`
2. Click `Create`
3. Name: `vfr-rg`
4. Region: choose the closest region available to your subscription

### 2. Create App Service

1. Open `App Services`
2. Click `Create`
3. Fill these values:
   - Resource Group: `vfr-rg`
   - Name: `virtual-fitting-room-web`
   - Publish: `Code`
   - Runtime stack: `.NET 6 (LTS)` if available
   - Operating System: `Windows`
   - Region: same region as the resource group
4. Create a new App Service Plan on the free or cheapest student-supported tier

### 3. Create Azure SQL

1. Open `SQL databases`
2. Click `Create`
3. Fill these values:
   - Database name: `VFR_DB`
   - Server: create a new SQL server
   - Authentication: SQL authentication
4. Save:
   - SQL server name
   - SQL username
   - SQL password
5. In the SQL server firewall, allow Azure services and your current client IP

## Phase 2 - Configure the website

Use these settings in Azure App Service > `Environment variables` or `Configuration`.

### Connection string

Name:

`ConnectionStrings__DefaultConnection`

Value:

`Server=tcp:YOUR_AZURE_SQL_SERVER.database.windows.net,1433;Initial Catalog=VFR_DB;Persist Security Info=False;User ID=YOUR_SQL_USER;Password=YOUR_SQL_PASSWORD;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;`

### AI settings

Add:

- `ASPNETCORE_ENVIRONMENT` = `Production`
- `VirtualTryOn__Mode` = `Replicate`
- `REPLICATE_API_TOKEN` = `r8_YOUR_REAL_REPLICATE_TOKEN`
- `VirtualTryOn__ReplicateVersion` = `cf5cb07a25e726fe2fac166a8c5ab52ddccd48657741670fb09d9954d4d8446f`
- `VirtualTryOn__ReplicatePersonFieldName` = `person_image`
- `VirtualTryOn__ReplicateClothingFieldName` = `cloth_image`
- `VirtualTryOn__ReplicateCategoryFieldName` = `cloth_type`

Alternative token setting:

- `VirtualTryOn__ReplicateApiToken` = `r8_YOUR_REAL_REPLICATE_TOKEN`

Use either `REPLICATE_API_TOKEN` or `VirtualTryOn__ReplicateApiToken`, not both.

## Phase 3 - Publish the ASP.NET MVC app

From Visual Studio:

1. Right click the project
2. Click `Publish`
3. Choose `Azure`
4. Choose `Azure App Service (Windows)`
5. Select `virtual-fitting-room-web`
6. Publish

Or from CLI:

`dotnet publish -c Release`

## Phase 4 - Database migration

After Azure SQL is ready:

1. Update the connection string locally if needed
2. Run Entity Framework migrations against Azure SQL
3. Confirm tables are created

## Phase 5 - Replicate AI

The web app calls Replicate directly from the server. Do not expose the Replicate token in frontend JavaScript or checked-in config files.

Recommended order:

1. First deploy the website only
2. Confirm login, pages, uploads, and DB work
3. Add `REPLICATE_API_TOKEN` in Azure App Service configuration
4. Restart the App Service
5. Test the try-on flow from a phone browser

## Phase 6 - Fallback plan if AI fails

If Replicate inference fails:

1. Confirm `REPLICATE_API_TOKEN` is configured in Azure
2. Confirm the Replicate account has billing/credit available
3. Check the returned Replicate HTTP error in the website error message
4. If the selected model is unavailable, replace `VirtualTryOn__ReplicateVersion` with another working Replicate model version

## Free local demo mode

For a no-payment local demo, use the built-in fast overlay server instead of Replicate:

- `VirtualTryOn:Mode` = `Local`
- `VirtualTryOn:ServerScriptPath` = `Scripts\fast_overlay_server.py`
- `VirtualTryOn:PythonExecutable` = `.venv-catvton\Scripts\python.exe`

This mode is free and runs on the local machine, but it is a lightweight garment overlay, not the full generative try-on model. It is suitable for demos without Replicate credit. To let users access it from anywhere, deploy the web app plus a reachable backend server; a local-only Python server on your laptop is not public internet hosting.

## Best execution order

1. Make Azure Web App work
2. Make Azure SQL work
3. Publish the ASP.NET site
4. Test the website without AI
5. Add the Replicate token
6. Test virtual try-on from a mobile network
