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
        private readonly UploadTryOnService _uploadTryOnService;
        private readonly IHttpClientFactory _httpClientFactory;

        public TryOnController(
            AppDbContext context,
            ImageProtectionService imageProtectionService,
            VirtualTryOnService virtualTryOnService,
            UploadTryOnService uploadTryOnService,
            IHttpClientFactory httpClientFactory)
        {
            _context = context;
            _imageProtectionService = imageProtectionService;
            _virtualTryOnService = virtualTryOnService;
            _uploadTryOnService = uploadTryOnService;
            _httpClientFactory = httpClientFactory;
        }

        [HttpGet]
        public async Task<IActionResult> Index()
        {
            int userId = HttpContext.Session.GetInt32("UserId") ?? 0;
            if (userId == 0)
                return RedirectToAction("Login", "Account");

            var user = await _context.UserMeasurements
                .AsNoTracking()
                .FirstOrDefaultAsync(u => u.Id == userId);

            if (user == null)
                return RedirectToAction("Login", "Account");

            ViewBag.ProfileComplete = user.Age > 0 &&
                user.Height > 0 &&
                user.Weight > 0 &&
                !string.IsNullOrWhiteSpace(user.Gender) &&
                !string.Equals(user.Gender, "Unspecified", StringComparison.OrdinalIgnoreCase);
            ViewBag.UserProfile = user;

            return View();
        }

        [HttpGet]
        public IActionResult Live(string? clothingImageUrl, string? clothingType, string? garmentArea)
        {
            ViewBag.IsPublicLive = true;
            ViewBag.InitialClothingUrl = clothingImageUrl?.Trim();
            ViewBag.InitialClothingType = string.IsNullOrWhiteSpace(clothingType) ? "T-Shirt" : clothingType.Trim();
            ViewBag.InitialGarmentArea = ResolveGarmentArea(clothingType, garmentArea) ?? "upper";

            return View();
        }

        [HttpGet]
        public IActionResult Upload()
        {
            int userId = HttpContext.Session.GetInt32("UserId") ?? 0;
            if (userId == 0)
                return RedirectToAction("Login", "Account");

            return View();
        }

        [HttpPost]
        public async Task<IActionResult> SaveProfile(string Gender, int Age, float Height, float Weight)
        {
            int userId = HttpContext.Session.GetInt32("UserId") ?? 0;
            if (userId == 0)
                return RedirectToAction("Login", "Account");

            var user = await _context.UserMeasurements.FirstOrDefaultAsync(u => u.Id == userId);
            if (user == null)
                return RedirectToAction("Login", "Account");

            if (string.IsNullOrWhiteSpace(Gender) || Age <= 0 || Height <= 0 || Weight <= 0)
            {
                ModelState.AddModelError("", "Please complete gender, age, height, and weight.");
                ViewBag.ProfileComplete = false;
                ViewBag.UserProfile = user;
                return View("Index");
            }

            user.Gender = Gender.Trim();
            user.Age = Age;
            user.Height = Height;
            user.Weight = Weight;
            await _context.SaveChangesAsync();

            return RedirectToAction(nameof(Index));
        }

        [HttpPost]
        public async Task<IActionResult> StartUploadSession(
            IFormFile? modelImageFile,
            IFormFile? garmentImageFile,
            string? modelImageData,
            string? modelImageUrl,
            string? garmentImageUrl,
            string? colabApiUrl,
            string? ClothingType,
            string? GarmentArea,
            CancellationToken cancellationToken = default)
        {
            int userId = HttpContext.Session.GetInt32("UserId") ?? 0;
            if (userId == 0)
                return RedirectToAction("Login", "Account");

            var modelImage = !string.IsNullOrWhiteSpace(modelImageData)
                ? ReadDataUrlImage(modelImageData, "Could not read the captured model photo.")
                : await ReadImageInputAsync(
                    modelImageFile,
                    modelImageUrl,
                    "Please upload a model image or capture one from the camera.",
                    cancellationToken);
            if (!modelImage.Success || modelImage.Bytes == null)
            {
                ModelState.AddModelError("", modelImage.Error ?? "Could not read the model image.");
                return View("Upload");
            }

            var garmentImage = await ReadImageInputAsync(
                garmentImageFile,
                garmentImageUrl,
                "Please upload a clothing image.",
                cancellationToken);
            if (!garmentImage.Success || garmentImage.Bytes == null)
            {
                ModelState.AddModelError("", garmentImage.Error ?? "Could not read the clothing image.");
                return View("Upload");
            }

            GarmentArea = ResolveGarmentArea(ClothingType, GarmentArea);
            if (string.IsNullOrWhiteSpace(GarmentArea))
            {
                ModelState.AddModelError("", "Please select the garment area.");
                return View("Upload");
            }

            var aiResult = await _uploadTryOnService.RunAsync(
                modelImage.Bytes,
                garmentImage.Bytes,
                GarmentArea,
                ClothingType,
                colabApiUrl,
                cancellationToken);

            if (!aiResult.Success || aiResult.OutputBytes == null)
            {
                ModelState.AddModelError("", aiResult.Error ?? "Virtual try-on failed.");
                return View("Upload");
            }

            var userImage = new UserImage
            {
                UserMeasurementId = userId,
                ImageData = _imageProtectionService.Protect(aiResult.OutputBytes),
                ImageType = "image/png",
                CreatedAt = DateTime.Now
            };

            _context.UserImages.Add(userImage);
            await _context.SaveChangesAsync(cancellationToken);

            return RedirectToAction("Measure", new { imageId = userImage.Id });
        }

        [HttpPost]
        public async Task<IActionResult> StartSession(
            IFormFile? imageFile,
            IFormFile? clothingFile,
            string? imageData,
            string? clothingImageData,
            string? poseImageUrl,
            string? clothingImageUrl,
            string? Gender,
            string? ClothingType,
            string? GarmentArea,
            CancellationToken cancellationToken = default
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
                await imageFile.CopyToAsync(ms, cancellationToken);
                finalImage = ms.ToArray();
                imageType = imageFile.ContentType;
            }
            else if (!string.IsNullOrWhiteSpace(poseImageUrl))
            {
                var downloadedImage = await DownloadImageFromUrlAsync(poseImageUrl, cancellationToken);
                if (!downloadedImage.Success || downloadedImage.Bytes == null)
                {
                    ModelState.AddModelError("", downloadedImage.Error ?? "Could not load the customer pose image URL.");
                    return await ReturnTryOnIndexAsync(userId);
                }

                finalImage = downloadedImage.Bytes;
                imageType = downloadedImage.ContentType ?? "image/png";
            }
            else
            {
                ModelState.AddModelError("", "Please provide a customer pose image");
                return await ReturnTryOnIndexAsync(userId);
            }

            if (clothingFile == null && string.IsNullOrWhiteSpace(clothingImageData) && string.IsNullOrWhiteSpace(clothingImageUrl))
            {
                ModelState.AddModelError("", "Please upload a clothing image");
                return await ReturnTryOnIndexAsync(userId);
            }

            GarmentArea = ResolveGarmentArea(ClothingType, GarmentArea);

            if (string.IsNullOrWhiteSpace(GarmentArea))
            {
                ModelState.AddModelError("", "Please select the garment area");
                return await ReturnTryOnIndexAsync(userId);
            }

            byte[] clothingBytes;
            if (!string.IsNullOrWhiteSpace(clothingImageData))
            {
                var base64 = clothingImageData.Split(',')[1];
                clothingBytes = Convert.FromBase64String(base64);
            }
            else if (!string.IsNullOrWhiteSpace(clothingImageUrl))
            {
                var downloadedClothing = await DownloadImageFromUrlAsync(clothingImageUrl, cancellationToken);
                if (!downloadedClothing.Success || downloadedClothing.Bytes == null)
                {
                    ModelState.AddModelError("", downloadedClothing.Error ?? "Could not load the clothing image URL.");
                    return await ReturnTryOnIndexAsync(userId);
                }

                clothingBytes = downloadedClothing.Bytes;
            }
            else
            {
                using var clothingStream = new MemoryStream();
                await clothingFile!.CopyToAsync(clothingStream, cancellationToken);
                clothingBytes = clothingStream.ToArray();
            }

            var aiResult = await _uploadTryOnService.RunAsync(
                finalImage,
                clothingBytes,
                GarmentArea,
                ClothingType,
                cancellationToken);
            if (!aiResult.Success || aiResult.OutputBytes == null)
            {
                ModelState.AddModelError("", aiResult.Error ?? "Virtual try-on failed.");
                return await ReturnTryOnIndexAsync(userId);
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

        private async Task<IActionResult> ReturnTryOnIndexAsync(int userId)
        {
            var user = await _context.UserMeasurements
                .AsNoTracking()
                .FirstOrDefaultAsync(u => u.Id == userId);

            if (user == null)
                return RedirectToAction("Login", "Account");

            ViewBag.ProfileComplete = user.Age > 0 &&
                user.Height > 0 &&
                user.Weight > 0 &&
                !string.IsNullOrWhiteSpace(user.Gender) &&
                !string.Equals(user.Gender, "Unspecified", StringComparison.OrdinalIgnoreCase);
            ViewBag.UserProfile = user;

            return View("Index");
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

            var imageBytes = _imageProtectionService.Unprotect(image.ImageData);
            ViewBag.ImageId = imageId;
            ViewBag.ImageUrl = $"data:{image.ImageType};base64,{Convert.ToBase64String(imageBytes)}";
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

        private static string? ResolveGarmentArea(string? clothingType, string? garmentArea)
        {
            var normalizedClothingType = clothingType?.Trim().ToLowerInvariant();
            return normalizedClothingType switch
            {
                "pants" => "lower",
                "shorts" => "lower",
                "short" => "lower",
                "dress" => "overall",
                "galabeya" => "overall",
                "galabiya" => "overall",
                "jellabiya" => "overall",
                "t-shirt" or "shirt" or "hoodie" or "jacket" => "upper",
                _ => string.IsNullOrWhiteSpace(garmentArea) ? garmentArea : garmentArea.Trim().ToLowerInvariant()
            };
        }

        private async Task<(bool Success, byte[]? Bytes, string? ContentType, string? Error)> DownloadImageFromUrlAsync(
            string imageUrl,
            CancellationToken cancellationToken)
        {
            const long maxBytes = 12 * 1024 * 1024;

            if (!Uri.TryCreate(imageUrl.Trim(), UriKind.Absolute, out var uri) ||
                uri.Scheme is not ("http" or "https"))
            {
                return (false, null, null, "Image URL must start with http:// or https://.");
            }

            using var client = _httpClientFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(30);

            using var response = await client.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            if (!response.IsSuccessStatusCode)
            {
                return (false, null, null, $"Image URL returned HTTP {(int)response.StatusCode}.");
            }

            var contentType = response.Content.Headers.ContentType?.MediaType;
            if (string.IsNullOrWhiteSpace(contentType) || !contentType.StartsWith("image/", StringComparison.OrdinalIgnoreCase))
            {
                return (false, null, null, "The URL did not return an image file.");
            }

            if (response.Content.Headers.ContentLength > maxBytes)
            {
                return (false, null, null, "Image URL is larger than 12 MB.");
            }

            var bytes = await response.Content.ReadAsByteArrayAsync(cancellationToken);
            return bytes.LongLength <= maxBytes
                ? (true, bytes, contentType, null)
                : (false, null, null, "Image URL is larger than 12 MB.");
        }

        private async Task<(bool Success, byte[]? Bytes, string? Error)> ReadImageInputAsync(
            IFormFile? imageFile,
            string? imageUrl,
            string missingMessage,
            CancellationToken cancellationToken)
        {
            if (imageFile != null && imageFile.Length > 0)
            {
                if (!string.IsNullOrWhiteSpace(imageFile.ContentType) &&
                    !imageFile.ContentType.StartsWith("image/", StringComparison.OrdinalIgnoreCase))
                {
                    return (false, null, "The selected file must be an image.");
                }

                using var stream = new MemoryStream();
                await imageFile.CopyToAsync(stream, cancellationToken);
                return (true, stream.ToArray(), null);
            }

            if (!string.IsNullOrWhiteSpace(imageUrl))
            {
                var downloadedImage = await DownloadImageFromUrlAsync(imageUrl, cancellationToken);
                return downloadedImage.Success && downloadedImage.Bytes != null
                    ? (true, downloadedImage.Bytes, null)
                    : (false, null, downloadedImage.Error);
            }

            return (false, null, missingMessage);
        }

        private static (bool Success, byte[]? Bytes, string? Error) ReadDataUrlImage(
            string imageData,
            string errorMessage)
        {
            try
            {
                var commaIndex = imageData.IndexOf(',');
                var base64 = commaIndex >= 0 ? imageData[(commaIndex + 1)..] : imageData;
                return (true, Convert.FromBase64String(base64), null);
            }
            catch
            {
                return (false, null, errorMessage);
            }
        }
    }
}
