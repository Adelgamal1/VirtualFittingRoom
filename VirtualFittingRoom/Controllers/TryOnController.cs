using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using VirtualFittingRoom.Data;
using VirtualFittingRoom.Models;
using VirtualFittingRoom.Services;

namespace VirtualFittingRoom.Controllers
{
    public class TryOnController : Controller
    {
        private readonly AppDbContext _context;
        private readonly ImageProtectionService _imageProtectionService;
        private readonly VirtualTryOnService _virtualTryOnService;

        public TryOnController(
            AppDbContext context,
            ImageProtectionService imageProtectionService,
            VirtualTryOnService virtualTryOnService)
        {
            _context = context;
            _imageProtectionService = imageProtectionService;
            _virtualTryOnService = virtualTryOnService;
        }

        [HttpGet]
        public IActionResult Index()
        {
            return RedirectToAction("Index", "Home");
        }

        [HttpPost]
        public async Task<IActionResult> StartSession(
            IFormFile? imageFile,
            IFormFile? clothingFile,
            string? imageData,
            string? clothingImageData,
            string? Gender,
            string? ClothingType,
            string? GarmentArea
        )
        {
            int userId = HttpContext.Session.GetInt32("UserId") ?? 0;
            if (userId == 0)
                return RedirectToAction("Login", "Account");

            byte[] finalImage;
            string imageType;

            if (!string.IsNullOrEmpty(imageData))
            {
                var base64 = imageData.Split(',')[1];
                finalImage = Convert.FromBase64String(base64);
                imageType = "image/png";
            }
            else if (imageFile != null && imageFile.Length > 0)
            {
                using var ms = new MemoryStream();
                await imageFile.CopyToAsync(ms);
                finalImage = ms.ToArray();
                imageType = imageFile.ContentType;
            }
            else
            {
                ModelState.AddModelError("", "Please provide a customer pose image");
                return View("~/Views/Home/Index.cshtml");
            }

            if (clothingFile == null && string.IsNullOrWhiteSpace(clothingImageData))
            {
                ModelState.AddModelError("", "Please upload a clothing image");
                return View("~/Views/Home/Index.cshtml");
            }

            if (string.IsNullOrWhiteSpace(GarmentArea))
            {
                ModelState.AddModelError("", "Please select the garment area");
                return View("~/Views/Home/Index.cshtml");
            }

            byte[] clothingBytes;
            if (!string.IsNullOrWhiteSpace(clothingImageData))
            {
                var base64 = clothingImageData.Split(',')[1];
                clothingBytes = Convert.FromBase64String(base64);
            }
            else
            {
                using var clothingStream = new MemoryStream();
                await clothingFile!.CopyToAsync(clothingStream);
                clothingBytes = clothingStream.ToArray();
            }

            var aiResult = await _virtualTryOnService.RunAsync(finalImage, clothingBytes, GarmentArea);
            if (!aiResult.Success || aiResult.OutputBytes == null)
            {
                ModelState.AddModelError("", aiResult.Error ?? "Virtual try-on failed.");
                return View("~/Views/Home/Index.cshtml");
            }

            var userImage = new UserImage
            {
                UserMeasurementId = userId,
                ImageData = _imageProtectionService.Protect(aiResult.OutputBytes),
                ImageType = "image/png",
                CreatedAt = DateTime.Now
            };

            _context.UserImages.Add(userImage);
            _context.SaveChanges();

            return RedirectToAction("Measure", new { imageId = userImage.Id });
        }

        [HttpGet]
        public IActionResult Measure(int imageId)
        {
            int userId = HttpContext.Session.GetInt32("UserId") ?? 0;
            if (userId == 0)
                return RedirectToAction("Login", "Account");

            var image = _context.UserImages
                .AsNoTracking()
                .FirstOrDefault(i => i.Id == imageId && i.UserMeasurementId == userId);

            if (image == null)
                return NotFound();

            ViewBag.ImageId = imageId;
            return View();
        }

        [HttpPost]
        public IActionResult Measure(int imageId, string Rating)
        {
            int userId = HttpContext.Session.GetInt32("UserId") ?? 0;
            if (userId == 0)
                return RedirectToAction("Login", "Account");

            var image = _context.UserImages
                .FirstOrDefault(i => i.Id == imageId && i.UserMeasurementId == userId);

            if (image == null)
                return NotFound();

            image.Rating = Rating;
            _context.SaveChanges();

            return RedirectToAction("History");
        }

        [HttpGet]
        public IActionResult History()
        {
            int userId = HttpContext.Session.GetInt32("UserId") ?? 0;
            if (userId == 0)
                return RedirectToAction("Login", "Account");

            var images = _context.UserImages
                .AsNoTracking()
                .Where(i => i.UserMeasurementId == userId)
                .OrderByDescending(i => i.CreatedAt)
                .ToList();

            var historyItems = images.Select(img => new HistoryItemViewModel
            {
                Id = img.Id,
                Rating = img.Rating,
                CreatedAt = img.CreatedAt,
                ImageType = img.ImageType,
                ImageUrl = $"data:{img.ImageType};base64,{Convert.ToBase64String(_imageProtectionService.Unprotect(img.ImageData))}"
            }).ToList();

            return View(historyItems);
        }

        [HttpGet]
        public IActionResult Download(int id)
        {
            int userId = HttpContext.Session.GetInt32("UserId") ?? 0;
            if (userId == 0)
                return RedirectToAction("Login", "Account");

            var img = _context.UserImages
                .AsNoTracking()
                .FirstOrDefault(i => i.Id == id && i.UserMeasurementId == userId);

            if (img == null)
                return NotFound();

            var imageBytes = _imageProtectionService.Unprotect(img.ImageData);
            return File(imageBytes, img.ImageType, $"fitting-session-{id}.png");
        }
    }
}
