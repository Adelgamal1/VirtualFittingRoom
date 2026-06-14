using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.EntityFrameworkCore;
using System;
using System.Linq;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;
using VirtualFittingRoom.Data;
using VirtualFittingRoom.Models;
using VirtualFittingRoom.Services;

namespace VirtualFittingRoom.Controllers
{
    public class AccountController : Controller
    {
        private readonly AppDbContext _context;
        private readonly IConfiguration _configuration;

        public AccountController(AppDbContext context, IConfiguration configuration)
        {
            _context = context;
            _configuration = configuration;
        }

        // ================= LOGIN =================
        [HttpGet]
        public IActionResult Login(string? externalError = null)
        {
            if (!string.IsNullOrWhiteSpace(externalError))
            {
                ModelState.AddModelError("", externalError);
            }

            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Login(string Email, string Password)
        {
            var normalizedEmail = (Email ?? string.Empty).Trim();
            var normalizedPassword = Password ?? string.Empty;

            var user = await _context.UserMeasurements
                .AsNoTracking()
                .FirstOrDefaultAsync(u => u.Email == normalizedEmail);

            if (user == null || user.PasswordHash != HashPassword(normalizedPassword))
            {
                ModelState.AddModelError("", "Invalid email or password");
                return View();
            }

            HttpContext.Session.SetInt32("UserId", user.Id);

            return RedirectToAction("Index", "TryOn");
        }

        // ================= SOCIAL LOGIN / SIGN UP =================
        [HttpGet]
        public IActionResult ExternalLogin(string provider, string returnUrl = "/TryOn")
        {
            var allowedProviders = new[] { "Google", "Facebook", "Apple" };
            var selectedProvider = allowedProviders
                .FirstOrDefault(p => string.Equals(p, provider, StringComparison.OrdinalIgnoreCase));

            if (selectedProvider == null)
            {
                ModelState.AddModelError("", "Unsupported sign-in provider");
                return View("Login");
            }

            if (!IsProviderConfigured(selectedProvider))
            {
                ModelState.AddModelError("", $"{selectedProvider} sign-in is not configured yet");
                return View("Login");
            }

            var redirectUrl = Url.Action(
                nameof(ExternalLoginCallback),
                "Account",
                new { returnUrl },
                Request.Scheme,
                Request.Host.ToString());
            var properties = new AuthenticationProperties { RedirectUri = redirectUrl };

            return Challenge(properties, selectedProvider);
        }

        [HttpGet]
        public async Task<IActionResult> ExternalLoginCallback(string returnUrl = "/TryOn")
        {
            var result = await HttpContext.AuthenticateAsync("External");

            if (!result.Succeeded || result.Principal == null)
            {
                ModelState.AddModelError("", "Social sign-in was not completed");
                return View("Login");
            }

            var email = result.Principal.FindFirstValue(ClaimTypes.Email);
            var fullName = result.Principal.FindFirstValue(ClaimTypes.Name)
                ?? result.Principal.FindFirstValue("name")
                ?? email;
            var providerName = result.Properties?.Items.TryGetValue(".AuthScheme", out var scheme) == true
                ? scheme
                : "Social";
            var providerUserId = result.Principal.FindFirstValue(ClaimTypes.NameIdentifier);

            if (string.IsNullOrWhiteSpace(email))
            {
                if (string.IsNullOrWhiteSpace(providerUserId))
                {
                    await HttpContext.SignOutAsync("External");
                    ModelState.AddModelError("", "The selected provider did not return enough account information");
                    return View("Login");
                }

                email = $"{providerName.ToLowerInvariant()}-{providerUserId}@social.local";
            }

            var normalizedEmail = email.Trim();
            var user = await _context.UserMeasurements
                .FirstOrDefaultAsync(u => u.Email == normalizedEmail);

            if (user == null)
            {
                user = new UserMeasurement
                {
                    FullName = fullName ?? normalizedEmail,
                    Email = normalizedEmail,
                    PhoneNumber = "Social login",
                    PasswordHash = HashPassword(Guid.NewGuid().ToString("N")),
                    Age = 18,
                    Weight = 0,
                    Height = 0,
                    Gender = "Unspecified"
                };

                _context.UserMeasurements.Add(user);
                await _context.SaveChangesAsync();
            }

            HttpContext.Session.SetInt32("UserId", user.Id);
            await HttpContext.SignOutAsync("External");

            return LocalRedirect(Url.IsLocalUrl(returnUrl) ? returnUrl : "/TryOn");
        }

        // ================= FORGOT PASSWORD =================
        [HttpGet]
        public IActionResult ForgotPassword()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> ForgotPassword(string Email)
        {
            var user = await _context.UserMeasurements.FirstOrDefaultAsync(u => u.Email == Email);

            if (user == null)
            {
                ViewBag.Message = "If email exists, reset link was sent";
                return View();
            }

            // 🔥 Generate secure token
            var token = Guid.NewGuid().ToString();

            user.ResetToken = token;
            user.ResetTokenExpiry = DateTime.Now.AddHours(1);

            await _context.SaveChangesAsync();

            // 🔥 Dynamic base URL (fix localhost problem)
            var request = HttpContext.Request;
            var baseUrl = $"{request.Scheme}://{request.Host}";

            var resetLink = $"{baseUrl}/Account/ResetPassword?token={token}";

            // 🔥 Send email (replace with your SMTP service)
            EmailService.Send(
                Email,
                "Virtual Fitting Room - Reset Password",
                $@"
                <h3>Password Reset Request</h3>
                <p>Click the link below:</p>
                <a href='{resetLink}'>Reset Password</a>
                <br/><br/>
                <p>If you didn't request this, ignore this email.</p>"
            );

            ViewBag.Message = "Reset link sent";
            return View();
        }

        // ================= RESET PASSWORD (GET) =================
        [HttpGet]
        public IActionResult ResetPassword(string token)
        {
            if (string.IsNullOrEmpty(token))
                return RedirectToAction("Login");

            ViewBag.Token = token;
            return View();
        }

        // ================= RESET PASSWORD (POST) =================
        [HttpPost]
        public async Task<IActionResult> ResetPassword(string token, string Password, string ConfirmPassword)
        {
            if (Password != ConfirmPassword)
            {
                ViewBag.Error = "Passwords do not match";
                return View();
            }

            var user = await _context.UserMeasurements
                .FirstOrDefaultAsync(x => x.ResetToken == token);

            if (user == null || user.ResetTokenExpiry < DateTime.Now)
            {
                ViewBag.Error = "Invalid or expired token";
                return View();
            }

            user.PasswordHash = HashPassword(Password);
            user.ResetToken = null;
            user.ResetTokenExpiry = null;

            await _context.SaveChangesAsync();

            return RedirectToAction("Login");
        }

        // ================= HASH PASSWORD =================
        private string HashPassword(string password)
        {
            using var sha256 = SHA256.Create();

            return Convert.ToBase64String(
                sha256.ComputeHash(Encoding.UTF8.GetBytes(password))
            );
        }

        private bool IsProviderConfigured(string provider)
        {
            return provider switch
            {
                "Google" => HasConfig("Authentication:Google:ClientId")
                    && HasConfig("Authentication:Google:ClientSecret"),
                "Facebook" => HasConfig("Authentication:Facebook:AppId")
                    && HasConfig("Authentication:Facebook:AppSecret"),
                "Apple" => HasConfig("Authentication:Apple:ClientId")
                    && HasConfig("Authentication:Apple:ClientSecret"),
                _ => false
            };
        }

        private bool HasConfig(string key)
        {
            return !string.IsNullOrWhiteSpace(_configuration[key]);
        }
    }
}
