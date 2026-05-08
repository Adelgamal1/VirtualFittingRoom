using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using VirtualFittingRoom.Data;
using VirtualFittingRoom.Services;

namespace VirtualFittingRoom.Controllers
{
    public class AccountController : Controller
    {
        private readonly AppDbContext _context;

        public AccountController(AppDbContext context)
        {
            _context = context;
        }

        // ================= LOGIN =================
        [HttpGet]
        public IActionResult Login()
        {
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

            return RedirectToAction("Index", "Home");
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
    }
}
