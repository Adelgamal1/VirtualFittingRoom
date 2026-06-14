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

            if (ViewBag.ProfileComplete)
                return RedirectToAction(nameof(Upload));

            return View();
        }

        [HttpGet]
        public IActionResult Live(string? clothingImageUrl, string? clothingType, string? garmentArea)
        {
            return RedirectToAction(nameof(Upload));
        }

        [HttpGet]
        public async Task<IActionResult> Upload()
        {
            int userId = HttpContext.Session.GetInt32("UserId") ?? 0;
            if (userId == 0)
                return RedirectToAction("Login", "Account");

            await PopulateUploadViewBagAsync(userId);
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
            string? Gender,
            string? ClothingType,
            string? GarmentArea,
            string? GarmentView,
            string? poseLandmarksData,
            CancellationToken cancellationToken = default)
        {
            int userId = HttpContext.Session.GetInt32("UserId") ?? 0;
            if (userId == 0)
                return RedirectToAction("Login", "Account");

            var targetBody = await ResolveTargetBodyAsync(userId, Gender, cancellationToken);
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
                return await ReturnUploadViewAsync(userId);
            }

            var garmentImage = await ReadImageInputAsync(
                garmentImageFile,
                garmentImageUrl,
                "Please upload a clothing image.",
                cancellationToken);
            if (!garmentImage.Success || garmentImage.Bytes == null)
            {
                ModelState.AddModelError("", garmentImage.Error ?? "Could not read the clothing image.");
                return await ReturnUploadViewAsync(userId);
            }

            GarmentArea = ResolveGarmentArea(ClothingType, GarmentArea);
            if (string.IsNullOrWhiteSpace(GarmentArea))
            {
                ModelState.AddModelError("", "Please select the garment area.");
                return await ReturnUploadViewAsync(userId);
            }

            if (!IsAllowedClothingForTarget(targetBody, ClothingType))
            {
                ModelState.AddModelError("", "Selected clothing category does not match the target body.");
                return await ReturnUploadViewAsync(userId);
            }

            var aiResult = await _uploadTryOnService.RunAsync(
                modelImage.Bytes,
                garmentImage.Bytes,
                GarmentArea,
                ClothingType,
                GarmentView,
                poseLandmarksData,
                cancellationToken);

            if (!aiResult.Success || aiResult.OutputBytes == null)
            {
                ModelState.AddModelError("", aiResult.Error ?? "Virtual try-on failed.");
                return await ReturnUploadViewAsync(userId);
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
            string? GarmentView,
            string? poseLandmarksData,
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

            var targetBody = await ResolveTargetBodyAsync(userId, Gender, cancellationToken);
            GarmentArea = ResolveGarmentArea(ClothingType, GarmentArea);

            if (string.IsNullOrWhiteSpace(GarmentArea))
            {
                ModelState.AddModelError("", "Please select the garment area");
                return await ReturnTryOnIndexAsync(userId);
            }

            if (!IsAllowedClothingForTarget(targetBody, ClothingType))
            {
                ModelState.AddModelError("", "Selected clothing category does not match the target body.");
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
                GarmentView,
                poseLandmarksData,
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

        private async Task<IActionResult> ReturnUploadViewAsync(int userId)
        {
            await PopulateUploadViewBagAsync(userId);
            return View("Upload");
        }

        private async Task PopulateUploadViewBagAsync(int userId)
        {
            ViewBag.UserProfile = await _context.UserMeasurements
                .AsNoTracking()
                .FirstOrDefaultAsync(u => u.Id == userId);
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

            var allowedRatings = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "Bad",
                "Good",
                "Very Good",
                "Excellent"
            };

            if (!allowedRatings.Contains(Rating ?? string.Empty))
            {
                return BadRequest("Invalid rating.");
            }

            var normalizedRating = allowedRatings.First(r =>
                string.Equals(r, Rating, StringComparison.OrdinalIgnoreCase));

            var image = _context.UserImages
                .FirstOrDefault(i => i.Id == imageId && i.UserMeasurementId == userId);

            if (image == null)
                return NotFound();

            image.Rating = normalizedRating;
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
                .OrderByDescending(i =>
                    i.Rating == "Excellent" ? 4 :
                    i.Rating == "Very Good" ? 3 :
                    i.Rating == "Good" ? 2 :
                    i.Rating == "Bad" ? 1 : 0)
                .ThenByDescending(i => i.CreatedAt)
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

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Delete(int id, CancellationToken cancellationToken = default)
        {
            int userId = HttpContext.Session.GetInt32("UserId") ?? 0;
            if (userId == 0)
                return RedirectToAction("Login", "Account");

            var img = await _context.UserImages
                .FirstOrDefaultAsync(i => i.Id == id && i.UserMeasurementId == userId, cancellationToken);

            if (img == null)
                return NotFound();

            _context.UserImages.Remove(img);
            await _context.SaveChangesAsync(cancellationToken);

            return RedirectToAction(nameof(History));
        }

        private static string? ResolveGarmentArea(string? clothingType, string? garmentArea)
        {
            var normalizedClothingType = NormalizeClothingType(clothingType);
            return normalizedClothingType switch
            {
                "pants" => "lower",
                "shorts" => "lower",
                "short" => "lower",
                "dress" => "overall",
                "jumpsuit" => "overall",
                "overall" => "overall",
                "romper" => "overall",
                "salopette" => "overall",
                "salopeit" => "overall",
                "سالوبيت" => "overall",
                "abaya" => "overall",
                "عباية" => "overall",
                "عبايات" => "overall",
                "galabeya" => "overall",
                "galabiya" => "overall",
                "jellabiya" => "overall",
                "t-shirt" or "jersey" or "tank-top" or "tanktop" or "shirt" or "chemise" or "blouse" or "hoodie" or "jacket" => "upper",
                _ => string.IsNullOrWhiteSpace(garmentArea) ? garmentArea : garmentArea.Trim().ToLowerInvariant()
            };
        }

        private async Task<string> ResolveTargetBodyAsync(int userId, string? postedGender, CancellationToken cancellationToken)
        {
            var normalized = NormalizeTargetBody(postedGender);
            if (!string.IsNullOrWhiteSpace(normalized))
            {
                return normalized;
            }

            var savedGender = await _context.UserMeasurements
                .AsNoTracking()
                .Where(u => u.Id == userId)
                .Select(u => u.Gender)
                .FirstOrDefaultAsync(cancellationToken);

            return NormalizeTargetBody(savedGender) ?? "male";
        }

        private static string? NormalizeTargetBody(string? value)
        {
            var normalized = (value ?? string.Empty).Trim().ToLowerInvariant();
            return normalized switch
            {
                "male" or "man" or "men" => "male",
                "female" or "woman" or "women" => "female",
                "child" or "children" or "kid" or "kids" => "child",
                _ => null
            };
        }

        private static bool IsAllowedClothingForTarget(string targetBody, string? clothingType)
        {
            var normalizedType = NormalizeClothingType(clothingType);
            if (string.IsNullOrWhiteSpace(normalizedType))
            {
                return true;
            }

            return targetBody switch
            {
                "female" => normalizedType is "t-shirt" or "shirt" or "chemise" or "blouse" or "hoodie" or "jacket" or "pants" or "shorts" or "dress" or "abaya" or "jumpsuit",
                "child" => normalizedType is "t-shirt" or "jersey" or "tank-top" or "shirt" or "hoodie" or "jacket" or "pants" or "shorts" or "dress" or "galabeya" or "jumpsuit",
                _ => normalizedType is "t-shirt" or "jersey" or "tank-top" or "shirt" or "hoodie" or "jacket" or "pants" or "shorts" or "galabeya"
            };
        }

        private static string NormalizeClothingType(string? clothingType)
        {
            var normalized = (clothingType ?? string.Empty)
                .Trim()
                .ToLowerInvariant()
                .Replace(" ", "-")
                .Replace("_", "-");

            return normalized switch
            {
                "tee" => "t-shirt",
                "tshirt" => "t-shirt",
                "t-shirts" => "t-shirt",
                "sports-shirt" => "jersey",
                "sport-shirt" => "jersey",
                "hockey-jersey" => "jersey",
                "football-jersey" => "jersey",
                "basketball-jersey" => "jersey",
                "tanktop" => "tank-top",
                "vest" => "tank-top",
                "sleeveless" => "tank-top",
                "sleeveless-shirt" => "tank-top",
                "chemise-shirt" => "chemise",
                "blouses" => "blouse",
                "hoodies" => "hoodie",
                "coats" => "jacket",
                "trouser" => "pants",
                "trousers" => "pants",
                "jeans" => "pants",
                "short" => "shorts",
                "galabiya" => "galabeya",
                "jellabiya" => "galabeya",
                "jalabiya" => "galabeya",
                "overall" => "jumpsuit",
                "overalls" => "jumpsuit",
                "romper" => "jumpsuit",
                "salopette" => "jumpsuit",
                "salopeit" => "jumpsuit",
                "سالوبيت" => "jumpsuit",
                "abayas" => "abaya",
                "عباية" => "abaya",
                "عبايات" => "abaya",
                _ => normalized
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
