using Microsoft.AspNetCore.Mvc;
using VirtualFittingRoom.Data;

namespace VirtualFittingRoom.Controllers
{
    public class HistoryController : Controller
    {
        private readonly AppDbContext _context;

        public HistoryController(AppDbContext context)
        {
            _context = context;
        }

        // ================= History Page =================
        public IActionResult Index()
        {
            // ✅ Check login
            var userId = HttpContext.Session.GetInt32("UserId");

            if (userId == null)
                return RedirectToAction("Login", "Account");

            // ✅ Get user images ordered by Rating then Date
            var images = _context.UserImages
                .Where(i => i.UserMeasurementId == userId)
                .OrderByDescending(i =>
                    i.Rating == "Excellent" ? 3 :
                    i.Rating == "Very Good" ? 2 : 1)
                .ThenByDescending(i => i.CreatedAt)
                .ToList();

            return View(images);
        }

        // ================= Download Image =================
        public IActionResult Download(int id)
        {
            var img = _context.UserImages.Find(id);
            if (img == null)
                return NotFound();

            return File(
                img.ImageData,
                img.ImageType,
                $"image_{id}.jpg"
            );
        }
    }
}
