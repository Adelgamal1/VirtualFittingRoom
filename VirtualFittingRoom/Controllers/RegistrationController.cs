using Microsoft.AspNetCore.Mvc;
using VirtualFittingRoom.Data;
using VirtualFittingRoom.Models;
using System.Security.Cryptography;
using System.Text;

namespace VirtualFittingRoom.Controllers
{
    public class RegistrationController : Controller
    {
        private readonly AppDbContext _context;

        public RegistrationController(AppDbContext context)
        {
            _context = context;
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public IActionResult Index(UserMeasurement model)
        {
            // 🔥 مهم جدًا
            ModelState.Remove("PasswordHash");

            if (!ModelState.IsValid)
                return View(model);

            // منع تكرار الإيميل
            if (_context.UserMeasurements.Any(u => u.Email == model.Email))
            {
                ModelState.AddModelError("Email", "Email already exists");
                return View(model);
            }

            // توليد Hash
            model.PasswordHash = HashPassword(model.Password);

            // عدم تخزين الباسورد plain
            model.Password = null;
            model.ConfirmPassword = null;

            _context.UserMeasurements.Add(model);
            _context.SaveChanges();

            return RedirectToAction("Login", "Account");
        }

        private string HashPassword(string password)
        {
            using var sha256 = SHA256.Create();
            var bytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(password));
            return Convert.ToBase64String(bytes);
        }
    }
}

