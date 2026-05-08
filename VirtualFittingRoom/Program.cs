using Microsoft.EntityFrameworkCore;
using Microsoft.Data.SqlClient;
using VirtualFittingRoom.Data;
using VirtualFittingRoom.Models;
using VirtualFittingRoom.Services;

var builder = WebApplication.CreateBuilder(args);

// Keep a fixed local URL only for development when no hosting URL is provided.
// Azure/App Service will inject its own binding via environment variables.
if (builder.Environment.IsDevelopment() &&
    string.IsNullOrWhiteSpace(builder.Configuration["ASPNETCORE_URLS"]))
{
    builder.WebHost.UseUrls("http://0.0.0.0:5000");
}

// ================= Services =================

builder.Services.AddControllersWithViews();
builder.Services.AddDataProtection();
builder.Services.AddHttpClient();
builder.Services.AddScoped<ImageProtectionService>();
builder.Services.Configure<VirtualTryOnOptions>(builder.Configuration.GetSection("VirtualTryOn"));
builder.Services.AddSingleton<LocalInferenceServerManager>();
builder.Services.AddScoped<VirtualTryOnService>();

builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlServer(
        builder.Configuration.GetConnectionString("DefaultConnection")));

builder.Services.AddSession(options =>
{
    options.IdleTimeout = TimeSpan.FromHours(2);
    options.Cookie.HttpOnly = true;
    options.Cookie.IsEssential = true;
});

var app = builder.Build();

await WarmUpDatabaseAsync(app);

// ================= Middleware =================

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
    app.UseHttpsRedirection();
}

app.UseStaticFiles();

app.UseRouting();

app.UseSession();

app.UseAuthorization();

// ================= Routes =================

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Account}/{action=Login}/{id?}");

app.Run();

static async Task WarmUpDatabaseAsync(WebApplication app)
{
    using var scope = app.Services.CreateScope();
    var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
    var logger = scope.ServiceProvider
        .GetRequiredService<ILoggerFactory>()
        .CreateLogger("DatabaseWarmUp");

    for (var attempt = 1; attempt <= 3; attempt++)
    {
        try
        {
            await db.Database.CanConnectAsync();
            return;
        }
        catch (SqlException ex) when (attempt < 3)
        {
            logger.LogWarning(ex, "Database connection failed during startup attempt {Attempt}. Retrying.", attempt);
            SqlConnection.ClearAllPools();
            await Task.Delay(TimeSpan.FromSeconds(2));
        }
    }
}
