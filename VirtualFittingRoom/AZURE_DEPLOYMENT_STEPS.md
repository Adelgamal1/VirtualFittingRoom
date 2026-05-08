# Azure Deployment Steps

## Goal

Deploy the ASP.NET MVC website to Azure App Service, move the database to Azure SQL, and keep AI inference as a separate API.

## Current architecture

- Website: ASP.NET MVC
- Database: currently local SQL/LocalDB
- AI: should be a separate API endpoint

## Important note

Do not run the heavy AI model inside Azure Web App.

Use this split:

1. Web App on Azure App Service
2. Database on Azure SQL
3. AI as external API

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

- `VirtualTryOn__Mode` = `Api`
- `VirtualTryOn__ApiUrl` = `https://YOUR-AI-API.azurewebsites.net/tryon`
- `VirtualTryOn__ApiKey` = leave empty unless your API uses a key
- `VirtualTryOn__ApiKeyHeader` = `Authorization`
- `VirtualTryOn__ApiPersonFieldName` = `person_image`
- `VirtualTryOn__ApiClothingFieldName` = `garment_image`
- `VirtualTryOn__ApiCategoryFieldName` = `category`
- `VirtualTryOn__ApiResponseImageField` = `outputImageBase64`

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

## Phase 5 - AI API

The web app should call an external AI API.

Recommended order:

1. First deploy the website only
2. Confirm login, pages, uploads, and DB work
3. Then connect the AI API

## Phase 6 - Fallback plan if AI fails

If the heavy model cannot run reliably on free infrastructure:

1. Replace the AI backend with a lighter ready model
2. Or switch to a lightweight body/garment overlay pipeline
3. Or retrain/adapt a smaller model later

## Best execution order

1. Make Azure Web App work
2. Make Azure SQL work
3. Publish the ASP.NET site
4. Test the website without AI
5. Add the AI API
6. If AI is unstable, replace it with a lighter model
