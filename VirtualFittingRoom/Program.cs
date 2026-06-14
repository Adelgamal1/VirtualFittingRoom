using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.HttpOverrides;
using Microsoft.EntityFrameworkCore;
using Microsoft.Data.SqlClient;
using VirtualFittingRoom.Data;
using VirtualFittingRoom.Models;
using VirtualFittingRoom.Services;

var builder = WebApplication.CreateBuilder(args);

var railwayPort = Environment.GetEnvironmentVariable("PORT");
if (!string.IsNullOrWhiteSpace(railwayPort) &&
    string.IsNullOrWhiteSpace(builder.Configuration["ASPNETCORE_URLS"]))
{
    builder.WebHost.UseUrls($"http://0.0.0.0:{railwayPort}");
}

// Keep a predictable local URL for Visual Studio and browser launch.
// Azure/App Service will inject its own binding via environment variables.
if (builder.Environment.IsDevelopment() &&
    string.IsNullOrWhiteSpace(builder.Configuration["ASPNETCORE_URLS"]))
{
    builder.WebHost.UseUrls("http://localhost:5001");
}

// ================= Services =================

builder.Services.AddControllersWithViews();
builder.Services.AddDataProtection();
builder.Services.AddHttpClient();
builder.Services.AddScoped<ImageProtectionService>();
builder.Services.Configure<VirtualTryOnOptions>(builder.Configuration.GetSection("VirtualTryOn"));
builder.Services.AddSingleton<LocalInferenceServerManager>();
builder.Services.AddScoped<VirtualTryOnService>();
builder.Services.AddScoped<UploadTryOnService>();

builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlServer(
        builder.Configuration.GetConnectionString("DefaultConnection")));

builder.Services.AddSession(options =>
{
    options.IdleTimeout = TimeSpan.FromHours(2);
    options.Cookie.HttpOnly = true;
    options.Cookie.IsEssential = true;
});

var authenticationBuilder = builder.Services
    .AddAuthentication(options =>
    {
        options.DefaultScheme = CookieAuthenticationDefaults.AuthenticationScheme;
    })
    .AddCookie()
    .AddCookie("External");

if (HasAuthConfig("Authentication:Google:ClientId") &&
    HasAuthConfig("Authentication:Google:ClientSecret"))
{
    authenticationBuilder.AddGoogle(options =>
    {
        options.ClientId = builder.Configuration["Authentication:Google:ClientId"] ?? string.Empty;
        options.ClientSecret = builder.Configuration["Authentication:Google:ClientSecret"] ?? string.Empty;
        options.CallbackPath = builder.Configuration["Authentication:Google:CallbackPath"] ?? "/signin-google";
        options.SignInScheme = "External";
        options.SaveTokens = true;
        options.CorrelationCookie.SameSite = SameSiteMode.Lax;
        options.CorrelationCookie.SecurePolicy = CookieSecurePolicy.SameAsRequest;
        options.Events.OnRemoteFailure = context =>
        {
            context.HandleResponse();
            context.Response.Redirect(BuildExternalErrorUrl("Google", context.Failure));
            return Task.CompletedTask;
        };
    });
}

if (HasAuthConfig("Authentication:Facebook:AppId") &&
    HasAuthConfig("Authentication:Facebook:AppSecret"))
{
    authenticationBuilder.AddFacebook(options =>
    {
        options.AppId = builder.Configuration["Authentication:Facebook:AppId"] ?? string.Empty;
        options.AppSecret = builder.Configuration["Authentication:Facebook:AppSecret"] ?? string.Empty;
        options.SignInScheme = "External";
        options.SaveTokens = true;
        options.CorrelationCookie.SameSite = SameSiteMode.Lax;
        options.CorrelationCookie.SecurePolicy = CookieSecurePolicy.SameAsRequest;
        options.Scope.Clear();
        options.Scope.Add("public_profile");
        options.Fields.Add("name");
        options.Events.OnRemoteFailure = context =>
        {
            context.HandleResponse();
            context.Response.Redirect(BuildExternalErrorUrl("Facebook", context.Failure));
            return Task.CompletedTask;
        };
    });
}

if (HasAuthConfig("Authentication:Apple:ClientId") &&
    HasAuthConfig("Authentication:Apple:ClientSecret"))
{
    authenticationBuilder.AddOpenIdConnect("Apple", "Apple", options =>
    {
        options.Authority = "https://appleid.apple.com";
        options.ClientId = builder.Configuration["Authentication:Apple:ClientId"] ?? string.Empty;
        options.ClientSecret = builder.Configuration["Authentication:Apple:ClientSecret"] ?? string.Empty;
        options.CallbackPath = "/signin-apple";
        options.ResponseType = "code";
        options.ResponseMode = "form_post";
        options.SignInScheme = "External";
        options.SaveTokens = true;
        options.CorrelationCookie.SameSite = SameSiteMode.Lax;
        options.CorrelationCookie.SecurePolicy = CookieSecurePolicy.SameAsRequest;
        options.NonceCookie.SameSite = SameSiteMode.Lax;
        options.NonceCookie.SecurePolicy = CookieSecurePolicy.SameAsRequest;
        options.Scope.Clear();
        options.Scope.Add("openid");
        options.Scope.Add("email");
        options.Scope.Add("name");
        options.Events.OnRemoteFailure = context =>
        {
            context.HandleResponse();
            context.Response.Redirect(BuildExternalErrorUrl("Apple", context.Failure));
            return Task.CompletedTask;
        };
    });
}

var app = builder.Build();

await WarmUpDatabaseAsync(app);

// ================= Middleware =================

app.UseForwardedHeaders(new ForwardedHeadersOptions
{
    ForwardedHeaders = ForwardedHeaders.XForwardedFor | ForwardedHeaders.XForwardedProto
});

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
    app.UseHttpsRedirection();
}

app.UseStaticFiles();

app.UseRouting();

app.UseSession();

app.UseAuthentication();
app.UseAuthorization();

// ================= Routes =================

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.MapGet("/healthz", () => Results.Ok("ok"));

app.Run();

static async Task WarmUpDatabaseAsync(WebApplication app)
{
    var logger = app.Services
        .GetRequiredService<ILoggerFactory>()
        .CreateLogger("DatabaseWarmUp");

    for (var attempt = 1; attempt <= 3; attempt++)
    {
        try
        {
            using var scope = app.Services.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
            await db.Database.CanConnectAsync();
            return;
        }
        catch (Exception ex) when (attempt < 3)
        {
            logger.LogWarning(ex, "Database connection failed during startup attempt {Attempt}. Retrying.", attempt);
            SqlConnection.ClearAllPools();
            await Task.Delay(TimeSpan.FromSeconds(2));
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "Database warm-up failed. The app will continue to start.");
        }
    }
}

bool HasAuthConfig(string key)
{
    return !string.IsNullOrWhiteSpace(builder.Configuration[key]);
}

string BuildExternalErrorUrl(string provider, Exception? failure)
{
    var detail = failure?.Message;
    var message = string.IsNullOrWhiteSpace(detail)
        ? $"{provider} sign-in failed. Please try again."
        : $"{provider} sign-in failed: {detail}";

    return $"/Account/Login?externalError={Uri.EscapeDataString(message)}";
}
