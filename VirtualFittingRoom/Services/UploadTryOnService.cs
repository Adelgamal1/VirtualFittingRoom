using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Collections.Generic;
using System.Text.Json;
using Microsoft.Extensions.Options;
using VirtualFittingRoom.Models;

namespace VirtualFittingRoom.Services
{
    public class UploadTryOnService
    {
        private readonly VirtualTryOnOptions _options;
        private readonly VirtualTryOnService _virtualTryOnService;

        public UploadTryOnService(
            IOptions<VirtualTryOnOptions> options,
            VirtualTryOnService virtualTryOnService)
        {
            _options = options.Value;
            _virtualTryOnService = virtualTryOnService;
        }

        public async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            CancellationToken cancellationToken = default)
        {
            return await RunAsync(personImage, clothingImage, garmentArea, null, cancellationToken);
        }

        public async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            string? clothingType,
            CancellationToken cancellationToken = default)
        {
            return await RunAsync(personImage, clothingImage, garmentArea, clothingType, null, null, cancellationToken);
        }

        public async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            string? clothingType,
            string? garmentView,
            CancellationToken cancellationToken = default)
        {
            return await RunAsync(personImage, clothingImage, garmentArea, clothingType, garmentView, null, null, cancellationToken);
        }

        public async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            string? clothingType,
            string? garmentView,
            string? poseLandmarksData,
            CancellationToken cancellationToken = default)
        {
            return await RunAsync(personImage, clothingImage, garmentArea, clothingType, garmentView, null, poseLandmarksData, cancellationToken);
        }

        public async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            string? clothingType,
            string? garmentView,
            string? apiUrlOverride,
            string? poseLandmarksData,
            CancellationToken cancellationToken = default)
        {
            if (ShouldUsePreviewEngine(apiUrlOverride))
            {
                try
                {
                    return (true, ComposeTryOnPreview(personImage, clothingImage, garmentArea, clothingType, poseLandmarksData), null);
                }
                catch (Exception ex)
                {
                    return (false, null, $"Upload try-on preview failed: {ex.Message}");
                }
            }

            var aiResult = !string.IsNullOrWhiteSpace(apiUrlOverride)
                ? await _virtualTryOnService.RunApiAsync(
                    personImage,
                    clothingImage,
                    garmentArea,
                    clothingType,
                    garmentView,
                    apiUrlOverride,
                    poseLandmarksData,
                    cancellationToken)
                : await _virtualTryOnService.RunAsync(
                    personImage,
                    clothingImage,
                    garmentArea,
                    clothingType,
                    garmentView,
                    poseLandmarksData,
                    cancellationToken);

            if (aiResult.Success)
            {
                return aiResult;
            }

            if (!ShouldUsePreviewFallback(aiResult.Error, apiUrlOverride))
            {
                return aiResult;
            }

            try
            {
                return (true, ComposeTryOnPreview(personImage, clothingImage, garmentArea, clothingType, poseLandmarksData), null);
            }
            catch (Exception ex)
            {
                return (false, null, $"Upload try-on failed: {ex.Message}");
            }
        }

        private bool ShouldUsePreviewEngine(string? apiUrlOverride)
        {
            var mode = (_options.Mode ?? string.Empty).Trim();
            return string.Equals(mode, "Preview", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(mode, "Fallback", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(mode, "Overlay", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(mode, "FastOverlay", StringComparison.OrdinalIgnoreCase);
        }

        private bool HasConfiguredCustomSpace()
        {
            var url = _options.HuggingFaceSpaceUrl?.Trim() ?? string.Empty;
            return url.Length > 0 &&
                !url.Contains("yisol-idm-vton", StringComparison.OrdinalIgnoreCase) &&
                !url.Contains("yisol/idm-vton", StringComparison.OrdinalIgnoreCase);
        }

        private bool ShouldUsePreviewFallback(string? error, string? apiUrlOverride)
        {
            var mode = (_options.Mode ?? string.Empty).Trim();
            if (string.Equals(mode, "Preview", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(mode, "Fallback", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            if (string.Equals(mode, "Api", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            if (IsRecoverableAiEndpointError(error) || IsExpiredCloudflareUrl(apiUrlOverride))
            {
                return true;
            }

            if (string.Equals(mode, "HuggingFace", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(mode, "HuggingFaceSpace", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            return false;
        }

        private static bool IsRecoverableAiEndpointError(string? error)
        {
            if (string.IsNullOrWhiteSpace(error))
            {
                return false;
            }

            return error.Contains("Colab API URL is missing", StringComparison.OrdinalIgnoreCase) ||
                error.Contains("Cloudflare tunnel", StringComparison.OrdinalIgnoreCase) ||
                error.Contains("No such host is known", StringComparison.OrdinalIgnoreCase) ||
                error.Contains("trycloudflare.com", StringComparison.OrdinalIgnoreCase) ||
                error.Contains("copying content to a stream", StringComparison.OrdinalIgnoreCase) ||
                error.Contains("response ended prematurely", StringComparison.OrdinalIgnoreCase) ||
                error.Contains("closed the connection", StringComparison.OrdinalIgnoreCase) ||
                error.Contains("forcibly closed", StringComparison.OrdinalIgnoreCase) ||
                error.Contains("connection reset", StringComparison.OrdinalIgnoreCase) ||
                error.Contains("Hugging Face Space URL is not configured", StringComparison.OrdinalIgnoreCase) ||
                error.Contains("old public demo", StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsExpiredCloudflareUrl(string? apiUrl)
        {
            return !string.IsNullOrWhiteSpace(apiUrl) &&
                apiUrl.Contains("trycloudflare.com", StringComparison.OrdinalIgnoreCase);
        }

        private static byte[] ComposeTryOnPreview(byte[] personBytes, byte[] garmentBytes, string garmentArea, string? clothingType, string? poseLandmarksData)
        {
            using var personInput = new MemoryStream(personBytes);
            using var garmentInput = new MemoryStream(garmentBytes);
            using var personSource = Image.FromStream(personInput);
            using var garmentSource = Image.FromStream(garmentInput);
            using var person = ResizeToMax(personSource, 1000);
            using var garment = ResizeToMax(garmentSource, 700);
            using var output = new Bitmap(person.Width, person.Height);

            using (var graphics = Graphics.FromImage(output))
            {
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.DrawImage(person, 0, 0, person.Width, person.Height);

                var category = NormalizeCategory(clothingType);
                var placement = ResolvePlacement(garmentArea, category);
                var body = DetectBodyFitLandmarks(person, poseLandmarksData);
                var target = BuildTargetRectangle(body, placement, category);
                using var preparedGarment = PrepareGarmentForPlacement(garment, category);
                var drawTarget = FitGarment(preparedGarment.Size, target, placement);
                DrawPreviewGarment(graphics, person, preparedGarment, drawTarget, body, placement, category);
            }

            using var result = new MemoryStream();
            output.Save(result, ImageFormat.Png);
            return result.ToArray();
        }

        private sealed record GarmentPlacement(
            string Area,
            double X,
            double Y,
            double Width,
            double Height,
            double FillHeight,
            double YBias);

        private sealed record BodyFitLandmarks(
            Rectangle Bounds,
            int CenterX,
            int NeckY,
            int CollarY,
            int ShoulderY,
            int ShoulderLeft,
            int ShoulderRight,
            int WaistY,
            int HipY,
            int FootY);

        private static Rectangle BuildTargetRectangle(BodyFitLandmarks body, GarmentPlacement placement, string category)
        {
            var reference = body.Bounds;
            if (placement.Area == "upper")
            {
                var shoulderWidth = Math.Max(1, body.ShoulderRight - body.ShoulderLeft);
                var widthFactor = category switch
                {
                    "tank-top" => 1.18,
                    "jersey" => 2.18,
                    "chemise" => 1.48,
                    "blouse" => 1.42,
                    "shirt" => 1.42,
                    "t-shirt" => 1.54,
                    "hoodie" or "jacket" => 1.58,
                    _ => 1.50
                };
                var width = ClampToImage((int)Math.Round(shoulderWidth * widthFactor), reference.Width, 8);
                var torsoHeight = Math.Max(1, body.HipY - body.NeckY);
                var height = category switch
                {
                    "jersey" => Math.Max(1, (int)Math.Round(torsoHeight * 1.40)),
                    "t-shirt" => Math.Max(1, (int)Math.Round(torsoHeight * 1.06)),
                    _ => Math.Max(1, (int)Math.Round(reference.Height * placement.Height))
                };
                var collarRatio = category switch
                {
                    "tank-top" => 0.13,
                    "jersey" => 0.12,
                    "t-shirt" => 0.115,
                    "hoodie" or "jacket" => 0.13,
                    _ => 0.12
                };
                var topLift = category switch
                {
                    "t-shirt" => shoulderWidth * 0.08,
                    "jersey" => shoulderWidth * 0.08,
                    _ => shoulderWidth * 0.06
                };
                var top = body.CollarY - (int)Math.Round(height * collarRatio) - (int)Math.Round(topLift);

                return ClampRectangle(
                    new Rectangle(body.CenterX - (width / 2), top, width, height),
                    reference);
            }

            if (placement.Area == "overall")
            {
                var shoulderWidth = Math.Max(1, body.ShoulderRight - body.ShoulderLeft);
                var widthFactor = category is "abaya" or "galabeya" ? 1.56 : 1.34;
                var width = ClampToImage((int)Math.Round(shoulderWidth * widthFactor), reference.Width, 10);
                var height = Math.Max(1, (int)Math.Round((body.FootY - body.NeckY) * 0.98));
                var top = body.CollarY - (int)Math.Round(reference.Height * 0.012);

                return ClampRectangle(
                    new Rectangle(body.CenterX - (width / 2), top, width, height),
                    reference);
            }

            if (placement.Area == "lower")
            {
                var width = Math.Max(1, (int)Math.Round(reference.Width * placement.Width));
                var height = Math.Max(1, (int)Math.Round((body.FootY - body.HipY) * 0.98));
                return ClampRectangle(
                    new Rectangle(body.CenterX - (width / 2), body.HipY - (int)Math.Round(reference.Height * 0.015), width, height),
                    reference);
            }

            return new Rectangle(
                reference.X + (int)Math.Round(reference.Width * placement.X),
                reference.Y + (int)Math.Round(reference.Height * placement.Y),
                (int)Math.Round(reference.Width * placement.Width),
                (int)Math.Round(reference.Height * placement.Height));
        }

        private static int ClampToImage(int preferred, int maximum, int margin)
        {
            return Math.Max(1, Math.Min(preferred, Math.Max(1, maximum - (margin * 2))));
        }

        private static Rectangle ClampRectangle(Rectangle rectangle, Rectangle bounds)
        {
            var x = Math.Max(0, rectangle.X);
            var y = Math.Max(0, rectangle.Y);
            var right = Math.Min(bounds.Right, rectangle.Right);
            var bottom = Math.Min(bounds.Bottom, rectangle.Bottom);

            if (right <= x)
            {
                right = Math.Min(bounds.Right, x + Math.Max(1, rectangle.Width));
            }

            if (bottom <= y)
            {
                bottom = Math.Min(bounds.Bottom, y + Math.Max(1, rectangle.Height));
            }

            return Rectangle.FromLTRB(x, y, right, bottom);
        }

        private static GarmentPlacement ResolvePlacement(string garmentArea, string? clothingType)
        {
            var category = NormalizeCategory(clothingType);
            var area = (garmentArea ?? "upper").Trim().ToLowerInvariant();

            return category switch
            {
                    "jersey" => new GarmentPlacement("upper", 0.02, 0.15, 0.96, 0.64, 1.00, 0.00),
                "hoodie" => new GarmentPlacement("upper", 0.08, 0.20, 0.84, 0.46, 1.00, 0.00),
                "jacket" => new GarmentPlacement("upper", 0.08, 0.20, 0.84, 0.46, 1.00, 0.00),
                "tank-top" => new GarmentPlacement("upper", 0.15, 0.20, 0.70, 0.46, 1.00, 0.00),
                "blouse" => new GarmentPlacement("upper", 0.10, 0.20, 0.80, 0.50, 1.00, 0.00),
                "chemise" => new GarmentPlacement("upper", 0.10, 0.20, 0.80, 0.50, 1.00, 0.00),
                "shirt" => new GarmentPlacement("upper", 0.12, 0.22, 0.76, 0.50, 1.00, 0.00),
                "t-shirt" => new GarmentPlacement("upper", 0.10, 0.20, 0.80, 0.50, 1.00, 0.00),
                "pants" => new GarmentPlacement("lower", 0.24, 0.45, 0.52, 0.52, 1.00, 0.00),
                "shorts" => new GarmentPlacement("lower", 0.26, 0.45, 0.48, 0.34, 1.00, 0.00),
                "dress" => new GarmentPlacement("overall", 0.17, 0.15, 0.66, 0.76, 1.00, 0.00),
                "abaya" => new GarmentPlacement("overall", 0.03, 0.18, 0.94, 0.80, 1.00, 0.00),
                "galabeya" => new GarmentPlacement("overall", 0.04, 0.19, 0.92, 0.78, 1.00, 0.00),
                "jumpsuit" => new GarmentPlacement("overall", 0.10, 0.14, 0.80, 0.80, 1.00, 0.00),
                _ => area switch
                {
                    "lower" => new GarmentPlacement("lower", 0.25, 0.45, 0.50, 0.50, 1.00, 0.00),
                    "overall" => new GarmentPlacement("overall", 0.08, 0.18, 0.84, 0.78, 1.00, 0.00),
                    _ => new GarmentPlacement("upper", 0.11, 0.22, 0.78, 0.40, 1.00, 0.00)
                }
            };
        }

        private static string NormalizeCategory(string? clothingType)
        {
            var normalized = (clothingType ?? string.Empty).Trim().ToLowerInvariant().Replace(" ", "-").Replace("_", "-");
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
                "abaya" => "abaya",
                "عباية" => "abaya",
                "عبايات" => "abaya",
                _ => normalized
            };
        }

        private static Rectangle FitGarment(Size sourceSize, Rectangle bounds, GarmentPlacement placement)
        {
            if (sourceSize.Width <= 0 || sourceSize.Height <= 0 || bounds.Width <= 0 || bounds.Height <= 0)
            {
                return bounds;
            }

            var sourceRatio = sourceSize.Width / (double)sourceSize.Height;
            var boundsRatio = bounds.Width / (double)bounds.Height;
            int width;
            int height;

            if (placement.Area == "upper")
            {
                height = Math.Max(1, (int)Math.Round(bounds.Height * placement.FillHeight));
                width = Math.Max(1, (int)Math.Round(height * sourceRatio));

                if (width > bounds.Width)
                {
                    width = bounds.Width;
                    height = Math.Max(1, (int)Math.Round(width / sourceRatio));
                }
            }
            else if (sourceRatio > boundsRatio)
            {
                width = bounds.Width;
                height = Math.Max(1, (int)Math.Round(width / sourceRatio));
            }
            else
            {
                height = bounds.Height;
                width = Math.Max(1, (int)Math.Round(height * sourceRatio));
            }

            return new Rectangle(
                bounds.X + ((bounds.Width - width) / 2),
                placement.Area == "upper"
                    ? bounds.Y + (int)Math.Round(bounds.Height * placement.YBias)
                    : bounds.Y + ((bounds.Height - height) / 2) + (int)Math.Round(bounds.Height * placement.YBias),
                width,
                height);
        }

        private static void DrawPreviewGarment(
            Graphics graphics,
            Bitmap person,
            Bitmap garment,
            Rectangle drawTarget,
            BodyFitLandmarks body,
            GarmentPlacement placement,
            string category)
        {
            if (placement.Area != "upper" || drawTarget.Width <= 0 || drawTarget.Height <= 0)
            {
                graphics.DrawImage(garment, drawTarget);
                return;
            }

            if (category == "jersey")
            {
                DrawTransparentUpperGarment(graphics, person, garment, drawTarget, body);
                return;
            }

            DrawUpperGarmentPreview(graphics, person, garment, drawTarget, body, category);
        }

        private static void DrawTransparentUpperGarment(
            Graphics graphics,
            Bitmap person,
            Bitmap garment,
            Rectangle drawTarget,
            BodyFitLandmarks body)
        {
            using var garmentLayer = new Bitmap(person.Width, person.Height, PixelFormat.Format32bppArgb);
            using (var layerGraphics = Graphics.FromImage(garmentLayer))
            {
                layerGraphics.CompositingQuality = CompositingQuality.HighQuality;
                layerGraphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                layerGraphics.SmoothingMode = SmoothingMode.HighQuality;
                layerGraphics.Clear(Color.Transparent);
                layerGraphics.DrawImage(garment, drawTarget);
            }

            var shoulderWidth = Math.Max(1, body.ShoulderRight - body.ShoulderLeft);
            var guardTop = Math.Max(0, body.NeckY - (int)Math.Round(shoulderWidth * 0.95));
            var guardBottom = Math.Min(person.Height, body.CollarY + (int)Math.Round(shoulderWidth * 0.18));
            using (var guardGraphics = Graphics.FromImage(garmentLayer))
            {
                guardGraphics.CompositingMode = CompositingMode.SourceCopy;
                using var clearBrush = new SolidBrush(Color.Transparent);
                guardGraphics.FillRectangle(clearBrush, 0, 0, person.Width, Math.Max(0, guardTop));

                var neckWidth = Math.Max(24, (int)Math.Round(shoulderWidth * 0.34));
                var neckHeight = Math.Max(14, (int)Math.Round(shoulderWidth * 0.18));
                guardGraphics.FillEllipse(
                    clearBrush,
                    body.CenterX - (neckWidth / 2),
                    Math.Max(0, body.CollarY - (neckHeight / 2)),
                    neckWidth,
                    neckHeight);
            }

            graphics.DrawImage(garmentLayer, 0, 0);
        }

        private static void DrawUpperGarmentPreview(
            Graphics graphics,
            Bitmap person,
            Bitmap garment,
            Rectangle drawTarget,
            BodyFitLandmarks body,
            string category)
        {
            var shoulderWidth = Math.Max(1, body.ShoulderRight - body.ShoulderLeft);
            var centerX = body.CenterX;
            var upperLift = category switch
            {
                "t-shirt" => shoulderWidth * 0.08,
                "jersey" => shoulderWidth * 0.08,
                _ => shoulderWidth * 0.06
            };
            var collarY = Math.Clamp(body.CollarY - (int)Math.Round(upperLift), 0, person.Height - 1);
            var shoulderY = Math.Clamp(body.ShoulderY - (int)Math.Round(upperLift * 0.70), 0, person.Height - 1);
            var hemY = category == "t-shirt"
                ? Math.Min(drawTarget.Bottom, body.WaistY + (int)Math.Round(shoulderWidth * 0.20))
                : Math.Min(drawTarget.Bottom, body.HipY + (int)Math.Round(shoulderWidth * 0.12));
            hemY = Math.Max(shoulderY + (int)Math.Round(shoulderWidth * 0.72), hemY);
            hemY = Math.Min(person.Height, hemY);

            var sleeveDrop = category switch
            {
                "tank-top" => 0.08,
                "jersey" => 0.26,
                "hoodie" or "jacket" => 0.22,
                "t-shirt" => 0.24,
                _ => 0.20
            };
            var sleeveReach = category switch
            {
                "tank-top" => 0.12,
                "jersey" => 0.56,
                "hoodie" or "jacket" => 0.48,
                "t-shirt" => 0.36,
                _ => 0.46
            };
            var bottomHalf = Math.Max(shoulderWidth * (category == "t-shirt" ? 0.45 : 0.58), drawTarget.Width * 0.31);
            var shoulderInset = shoulderWidth * (category == "t-shirt" ? 0.03f : 0.12f);
            var leftOuter = Math.Max(drawTarget.Left, (float)body.ShoulderLeft - (shoulderWidth * (float)sleeveReach));
            var rightOuter = Math.Min(drawTarget.Right, (float)body.ShoulderRight + (shoulderWidth * (float)sleeveReach));
            var outerY = shoulderY + (shoulderWidth * (float)sleeveDrop);
            var leftHem = Math.Max(drawTarget.Left, centerX - (float)bottomHalf);
            var rightHem = Math.Min(drawTarget.Right, centerX + (float)bottomHalf);
            var waistY = Math.Min(hemY - 1, Math.Max(shoulderY + shoulderWidth * 0.55f, body.WaistY));
            var sideCurve = shoulderWidth * 0.18f;
            var sleeveCurve = shoulderWidth * 0.08f;

            using var upperPath = new GraphicsPath();
            upperPath.StartFigure();
            upperPath.AddBezier(
                leftOuter,
                outerY,
                leftOuter + sleeveCurve,
                shoulderY + (shoulderWidth * 0.08f),
                body.ShoulderLeft - shoulderInset - sleeveCurve,
                collarY + (shoulderWidth * 0.02f),
                body.ShoulderLeft - shoulderInset,
                collarY);
            upperPath.AddBezier(
                body.ShoulderLeft - shoulderInset,
                collarY,
                centerX - (shoulderWidth * 0.22f),
                collarY - (shoulderWidth * 0.06f),
                centerX + (shoulderWidth * 0.22f),
                collarY - (shoulderWidth * 0.06f),
                body.ShoulderRight + shoulderInset,
                collarY);
            upperPath.AddBezier(
                body.ShoulderRight + shoulderInset,
                collarY,
                body.ShoulderRight + shoulderInset + sleeveCurve,
                collarY + (shoulderWidth * 0.02f),
                rightOuter - sleeveCurve,
                shoulderY + (shoulderWidth * 0.08f),
                rightOuter,
                outerY);
            upperPath.AddBezier(
                rightOuter,
                outerY,
                rightOuter - sideCurve,
                waistY,
                rightHem + (sideCurve * 0.30f),
                hemY,
                rightHem,
                hemY);
            upperPath.AddBezier(
                rightHem,
                hemY,
                centerX + (shoulderWidth * 0.24f),
                hemY + (shoulderWidth * 0.025f),
                centerX - (shoulderWidth * 0.24f),
                hemY + (shoulderWidth * 0.025f),
                leftHem,
                hemY);
            upperPath.AddBezier(
                leftHem,
                hemY,
                leftHem - (sideCurve * 0.30f),
                hemY,
                leftOuter + sideCurve,
                waistY,
                leftOuter,
                outerY);
            upperPath.CloseFigure();

            var neckCenterY = Math.Clamp(
                collarY + (int)Math.Round(shoulderWidth * (category == "t-shirt" ? 0.055 : 0.035)),
                0,
                person.Height - 1);
            var neckWidth = Math.Max(22, shoulderWidth * (category == "tank-top" ? 0.28f : category == "t-shirt" ? 0.26f : 0.23f));
            var neckHeight = Math.Max(11, shoulderWidth * (category == "tank-top" ? 0.12f : category == "t-shirt" ? 0.095f : 0.105f));
            var neckRect = new RectangleF(
                centerX - (neckWidth / 2f),
                neckCenterY - (neckHeight * 0.50f),
                neckWidth,
                neckHeight);

            using var neckPath = new GraphicsPath();
            if (category == "t-shirt")
            {
                var neckLeft = neckRect.Left;
                var neckRight = neckRect.Right;
                var neckTop = neckRect.Top + (neckRect.Height * 0.10f);
                var neckBottom = neckRect.Bottom;
                neckPath.StartFigure();
                neckPath.AddBezier(
                    neckLeft,
                    neckTop + (neckRect.Height * 0.30f),
                    centerX - (neckWidth * 0.42f),
                    neckTop - (neckRect.Height * 0.10f),
                    centerX + (neckWidth * 0.42f),
                    neckTop - (neckRect.Height * 0.10f),
                    neckRight,
                    neckTop + (neckRect.Height * 0.30f));
                neckPath.AddBezier(
                    neckRight,
                    neckTop + (neckRect.Height * 0.30f),
                    centerX + (neckWidth * 0.34f),
                    neckBottom,
                    centerX - (neckWidth * 0.34f),
                    neckBottom,
                    neckLeft,
                    neckTop + (neckRect.Height * 0.30f));
                neckPath.CloseFigure();
            }
            else
            {
                neckPath.AddEllipse(neckRect);
            }

            using var garmentLayer = new Bitmap(person.Width, person.Height, PixelFormat.Format32bppArgb);
            using (var layerGraphics = Graphics.FromImage(garmentLayer))
            {
                layerGraphics.CompositingQuality = CompositingQuality.HighQuality;
                layerGraphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                layerGraphics.SmoothingMode = SmoothingMode.HighQuality;
                layerGraphics.Clear(Color.Transparent);

                using var garmentRegion = new Region(upperPath);
                garmentRegion.Exclude(neckPath);
                layerGraphics.SetClip(garmentRegion, CombineMode.Replace);
                layerGraphics.DrawImage(garment, drawTarget);
                layerGraphics.ResetClip();

                using var sideShade = new LinearGradientBrush(
                    new Rectangle(Math.Max(0, drawTarget.Left), Math.Max(0, drawTarget.Top), Math.Max(1, drawTarget.Width), Math.Max(1, drawTarget.Height)),
                    Color.FromArgb(46, 0, 0, 0),
                    Color.FromArgb(4, 255, 255, 255),
                    LinearGradientMode.Horizontal);
                using var shadeRegion = new Region(upperPath);
                shadeRegion.Exclude(neckPath);
                layerGraphics.SetClip(shadeRegion, CombineMode.Replace);
                layerGraphics.CompositingMode = CompositingMode.SourceOver;
                layerGraphics.FillRectangle(sideShade, drawTarget);
                layerGraphics.ResetClip();
            }

            graphics.DrawImage(garmentLayer, 0, 0);

            var skinColor = EstimateSkinColor(person, centerX, Math.Max(0, body.NeckY - (int)Math.Round(shoulderWidth * 0.10)), shoulderWidth);
            using (var neckBrush = new LinearGradientBrush(
                neckRect,
                ShadeColor(skinColor, 1.08f),
                ShadeColor(skinColor, 0.94f),
                LinearGradientMode.Vertical))
            {
                graphics.FillPath(neckBrush, neckPath);
            }

            var collarColor = EstimateGarmentTrimColor(garment);
            using var collarPen = new Pen(Color.FromArgb(215, collarColor), Math.Max(1.8f, shoulderWidth * 0.020f));
            using var collarShadow = new Pen(Color.FromArgb(58, 0, 0, 0), Math.Max(2.2f, shoulderWidth * 0.028f));
            var collarArc = new RectangleF(
                centerX - (neckWidth * 0.58f),
                neckCenterY - (neckHeight * 0.48f),
                neckWidth * 1.16f,
                neckHeight * 1.18f);
            graphics.DrawArc(collarShadow, collarArc, 178, 184);
            graphics.DrawArc(collarPen, collarArc, 178, 184);

            using var hemShadow = new Pen(Color.FromArgb(48, 0, 0, 0), Math.Max(1.2f, shoulderWidth * 0.01f));
            graphics.DrawLine(hemShadow, leftHem + 4, hemY - 1, rightHem - 4, hemY - 1);
        }

        private static Color EstimateSkinColor(Bitmap person, int centerX, int centerY, int shoulderWidth)
        {
            long r = 0;
            long g = 0;
            long b = 0;
            var count = 0;
            var radiusX = Math.Max(8, (int)Math.Round(shoulderWidth * 0.08));
            var radiusY = Math.Max(8, (int)Math.Round(shoulderWidth * 0.08));
            var fromX = Math.Max(0, centerX - radiusX);
            var toX = Math.Min(person.Width - 1, centerX + radiusX);
            var fromY = Math.Max(0, centerY - radiusY);
            var toY = Math.Min(person.Height - 1, centerY + radiusY);

            for (var y = fromY; y <= toY; y += 2)
            {
                for (var x = fromX; x <= toX; x += 2)
                {
                    var pixel = person.GetPixel(x, y);
                    var brightest = Math.Max(pixel.R, Math.Max(pixel.G, pixel.B));
                    var darkest = Math.Min(pixel.R, Math.Min(pixel.G, pixel.B));
                    if (brightest < 35 || brightest - darkest < 8)
                    {
                        continue;
                    }

                    r += pixel.R;
                    g += pixel.G;
                    b += pixel.B;
                    count++;
                }
            }

            return count == 0
                ? Color.FromArgb(150, 92, 62)
                : Color.FromArgb((int)(r / count), (int)(g / count), (int)(b / count));
        }

        private static Color ShadeColor(Color color, float factor)
        {
            return Color.FromArgb(
                color.A,
                Math.Clamp((int)Math.Round(color.R * factor), 0, 255),
                Math.Clamp((int)Math.Round(color.G * factor), 0, 255),
                Math.Clamp((int)Math.Round(color.B * factor), 0, 255));
        }

        private static Color EstimateGarmentTrimColor(Bitmap garment)
        {
            long r = 0;
            long g = 0;
            long b = 0;
            var count = 0;
            var fromX = Math.Max(0, (int)Math.Round(garment.Width * 0.36));
            var toX = Math.Min(garment.Width, (int)Math.Round(garment.Width * 0.64));
            var fromY = Math.Max(0, (int)Math.Round(garment.Height * 0.04));
            var toY = Math.Min(garment.Height, (int)Math.Round(garment.Height * 0.22));

            for (var y = fromY; y < toY; y += 2)
            {
                for (var x = fromX; x < toX; x += 2)
                {
                    var pixel = garment.GetPixel(x, y);
                    if (pixel.A < 32 || IsBrightBackground(pixel))
                    {
                        continue;
                    }

                    r += pixel.R;
                    g += pixel.G;
                    b += pixel.B;
                    count++;
                }
            }

            if (count == 0)
            {
                return Color.FromArgb(24, 24, 27);
            }

            return Color.FromArgb(
                (int)(r / count),
                (int)(g / count),
                (int)(b / count));
        }

        private static Bitmap ResizeToMax(Image source, int maxSide)
        {
            var largestSide = Math.Max(source.Width, source.Height);
            var scale = largestSide > maxSide ? maxSide / (double)largestSide : 1.0;
            var width = Math.Max(1, (int)Math.Round(source.Width * scale));
            var height = Math.Max(1, (int)Math.Round(source.Height * scale));
            var bitmap = new Bitmap(width, height);

            using var graphics = Graphics.FromImage(bitmap);
            graphics.CompositingQuality = CompositingQuality.HighQuality;
            graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
            graphics.SmoothingMode = SmoothingMode.HighQuality;
            graphics.DrawImage(source, 0, 0, width, height);
            return bitmap;
        }

        private static Rectangle GetPersonReferenceBounds(Bitmap person)
        {
            var detected = DetectForegroundBounds(person);
            if (detected.HasValue)
            {
                return detected.Value;
            }

            var width = Math.Max(1, (int)Math.Round(person.Width * 0.64));
            var x = Math.Max(0, (person.Width - width) / 2);
            return new Rectangle(x, 0, width, person.Height);
        }

        private static BodyFitLandmarks DetectBodyFitLandmarks(Bitmap person, string? poseLandmarksData)
        {
            var bounds = GetPersonReferenceBounds(person);
            var fromPose = TryDetectBodyFitLandmarksFromPose(person, bounds, poseLandmarksData);
            if (fromPose != null)
            {
                return fromPose;
            }

            var background = EstimateBackgroundColor(person);
            var rowSpans = BuildForegroundRowSpans(person, background, bounds);
            var centerX = FindMedianCenterX(rowSpans, bounds) ?? (bounds.X + (bounds.Width / 2));
            var shoulderY = EstimateShoulderY(rowSpans, bounds);
            var shoulderSpan = FindBestSpanNear(rowSpans, bounds, shoulderY, Math.Max(4, bounds.Height / 32));
            var shoulderWidth = Math.Max(1, shoulderSpan.Right - shoulderSpan.Left);

            if (shoulderWidth < bounds.Width * 0.28)
            {
                shoulderWidth = (int)Math.Round(bounds.Width * 0.42);
                shoulderSpan = new RowSpan(centerX - (shoulderWidth / 2), centerX + (shoulderWidth / 2));
            }

            var waistY = bounds.Y + (int)Math.Round(bounds.Height * 0.47);
            var hipY = bounds.Y + (int)Math.Round(bounds.Height * 0.55);
            var footY = bounds.Bottom - Math.Max(2, bounds.Height / 90);

            return new BodyFitLandmarks(
                bounds,
                centerX,
                shoulderY - (int)Math.Round(shoulderWidth * 0.18),
                shoulderY - (int)Math.Round(shoulderWidth * 0.07),
                shoulderY,
                Math.Max(bounds.X, shoulderSpan.Left),
                Math.Min(bounds.Right, shoulderSpan.Right),
                waistY,
                hipY,
                footY);
        }

        private sealed record PosePoint(double X, double Y, double Visibility);

        private static BodyFitLandmarks? TryDetectBodyFitLandmarksFromPose(Bitmap person, Rectangle foregroundBounds, string? poseLandmarksData)
        {
            if (string.IsNullOrWhiteSpace(poseLandmarksData))
            {
                return null;
            }

            try
            {
                using var document = JsonDocument.Parse(poseLandmarksData);
                if (!document.RootElement.TryGetProperty("landmarks", out var landmarks) ||
                    landmarks.ValueKind != JsonValueKind.Array)
                {
                    return null;
                }

                PosePoint? ReadPoint(int index, double minVisibility = 0.24)
                {
                    if (landmarks.GetArrayLength() <= index)
                    {
                        return null;
                    }

                    var item = landmarks[index];
                    if (item.ValueKind != JsonValueKind.Object ||
                        !item.TryGetProperty("x", out var xNode) ||
                        !item.TryGetProperty("y", out var yNode))
                    {
                        return null;
                    }

                    var visibility = item.TryGetProperty("visibility", out var visibilityNode)
                        ? visibilityNode.GetDouble()
                        : 1.0;

                    if (visibility < minVisibility)
                    {
                        return null;
                    }

                    var x = xNode.GetDouble();
                    var y = yNode.GetDouble();
                    if (x <= 1.5 && y <= 1.5)
                    {
                        x *= person.Width;
                        y *= person.Height;
                    }

                    if (x < -person.Width * 0.1 || x > person.Width * 1.1 ||
                        y < -person.Height * 0.1 || y > person.Height * 1.1)
                    {
                        return null;
                    }

                    return new PosePoint(x, y, visibility);
                }

                var shoulderA = ReadPoint(11, 0.35);
                var shoulderB = ReadPoint(12, 0.35);
                var hipA = ReadPoint(23, 0.25);
                var hipB = ReadPoint(24, 0.25);
                if (shoulderA == null || shoulderB == null || hipA == null || hipB == null)
                {
                    return null;
                }

                var leftShoulder = shoulderA.X <= shoulderB.X ? shoulderA : shoulderB;
                var rightShoulder = shoulderA.X <= shoulderB.X ? shoulderB : shoulderA;
                var leftHip = hipA.X <= hipB.X ? hipA : hipB;
                var rightHip = hipA.X <= hipB.X ? hipB : hipA;
                var shoulderDistance = Math.Abs(rightShoulder.X - leftShoulder.X);
                var hipDistance = Math.Abs(rightHip.X - leftHip.X);
                if (shoulderDistance < person.Width * 0.08 || shoulderDistance > person.Width * 0.75)
                {
                    return null;
                }

                var shoulderCenterX = (leftShoulder.X + rightShoulder.X) / 2.0;
                var shoulderCenterY = (leftShoulder.Y + rightShoulder.Y) / 2.0;
                var hipCenterX = (leftHip.X + rightHip.X) / 2.0;
                var hipCenterY = (leftHip.Y + rightHip.Y) / 2.0;
                var shoulderFromTop = (shoulderCenterY - foregroundBounds.Top) / Math.Max(1.0, foregroundBounds.Height);
                var hipFromTop = (hipCenterY - foregroundBounds.Top) / Math.Max(1.0, foregroundBounds.Height);
                if (shoulderFromTop < 0.12 ||
                    shoulderFromTop > 0.39 ||
                    hipFromTop < 0.40 ||
                    hipFromTop > 0.76 ||
                    hipCenterY - shoulderCenterY < foregroundBounds.Height * 0.18)
                {
                    return null;
                }

                var centerX = (int)Math.Round((shoulderCenterX * 0.58) + (hipCenterX * 0.42));
                var nose = ReadPoint(0, 0.20);
                var rawNeckY = nose != null
                    ? (int)Math.Round(shoulderCenterY - ((shoulderCenterY - nose.Y) * 0.18))
                    : (int)Math.Round(shoulderCenterY - (shoulderDistance * 0.20));
                var collarY = (int)Math.Round(shoulderCenterY - (shoulderDistance * 0.08));
                var neckY = Math.Min(rawNeckY, collarY - (int)Math.Round(shoulderDistance * 0.025));

                var ankleA = ReadPoint(27, 0.12) ?? ReadPoint(29, 0.12) ?? ReadPoint(31, 0.12);
                var ankleB = ReadPoint(28, 0.12) ?? ReadPoint(30, 0.12) ?? ReadPoint(32, 0.12);
                var footY = new[] { ankleA?.Y, ankleB?.Y, (double?)foregroundBounds.Bottom }
                    .Where(value => value.HasValue)
                    .Select(value => value!.Value)
                    .DefaultIfEmpty(foregroundBounds.Bottom)
                    .Max();

                var poseMinX = new[]
                    {
                        leftShoulder.X,
                        rightShoulder.X,
                        leftHip.X,
                        rightHip.X,
                        ReadPoint(13, 0.12)?.X,
                        ReadPoint(14, 0.12)?.X,
                        ReadPoint(15, 0.12)?.X,
                        ReadPoint(16, 0.12)?.X
                    }
                    .Where(value => value.HasValue)
                    .Select(value => value!.Value)
                    .DefaultIfEmpty(foregroundBounds.Left)
                    .Min();

                var poseMaxX = new[]
                    {
                        leftShoulder.X,
                        rightShoulder.X,
                        leftHip.X,
                        rightHip.X,
                        ReadPoint(13, 0.12)?.X,
                        ReadPoint(14, 0.12)?.X,
                        ReadPoint(15, 0.12)?.X,
                        ReadPoint(16, 0.12)?.X
                    }
                    .Where(value => value.HasValue)
                    .Select(value => value!.Value)
                    .DefaultIfEmpty(foregroundBounds.Right)
                    .Max();

                var marginX = Math.Max(12, (int)Math.Round(Math.Max(shoulderDistance, hipDistance) * 0.42));
                var marginTop = Math.Max(10, (int)Math.Round(shoulderDistance * 0.36));
                var bounds = Rectangle.FromLTRB(
                    Math.Max(0, Math.Min(foregroundBounds.Left, (int)Math.Round(poseMinX) - marginX)),
                    Math.Max(0, Math.Min(foregroundBounds.Top, neckY - marginTop)),
                    Math.Min(person.Width, Math.Max(foregroundBounds.Right, (int)Math.Round(poseMaxX) + marginX)),
                    Math.Min(person.Height, Math.Max(foregroundBounds.Bottom, (int)Math.Round(footY))));

                return new BodyFitLandmarks(
                    bounds,
                    Math.Clamp(centerX, 0, person.Width - 1),
                    Math.Clamp(neckY, 0, person.Height - 1),
                    Math.Clamp(collarY, 0, person.Height - 1),
                    Math.Clamp((int)Math.Round(shoulderCenterY), 0, person.Height - 1),
                    Math.Clamp((int)Math.Round(leftShoulder.X), 0, person.Width - 1),
                    Math.Clamp((int)Math.Round(rightShoulder.X), 0, person.Width - 1),
                    Math.Clamp((int)Math.Round(hipCenterY - ((hipCenterY - shoulderCenterY) * 0.20)), 0, person.Height - 1),
                    Math.Clamp((int)Math.Round(hipCenterY), 0, person.Height - 1),
                    Math.Clamp((int)Math.Round(footY), 0, person.Height - 1));
            }
            catch
            {
                return null;
            }
        }

        private readonly record struct RowSpan(int Left, int Right)
        {
            public int Width => Math.Max(0, Right - Left);
        }

        private static RowSpan?[] BuildForegroundRowSpans(Bitmap image, Color background, Rectangle bounds)
        {
            var spans = new RowSpan?[image.Height];
            for (var y = bounds.Top; y < bounds.Bottom; y++)
            {
                var left = image.Width;
                var right = -1;
                for (var x = bounds.Left; x < bounds.Right; x += 2)
                {
                    var pixel = image.GetPixel(x, y);
                    if (ColorDistance(pixel, background) <= 42)
                    {
                        continue;
                    }

                    left = Math.Min(left, x);
                    right = Math.Max(right, x);
                }

                if (right > left)
                {
                    spans[y] = new RowSpan(left, Math.Min(image.Width, right + 2));
                }
            }

            return spans;
        }

        private static int? FindMedianCenterX(RowSpan?[] spans, Rectangle bounds)
        {
            var centers = new List<int>();
            var from = bounds.Y + (int)Math.Round(bounds.Height * 0.22);
            var to = bounds.Y + (int)Math.Round(bounds.Height * 0.62);

            for (var y = Math.Max(0, from); y < Math.Min(spans.Length, to); y++)
            {
                if (spans[y] is not RowSpan span || span.Width < bounds.Width * 0.16)
                {
                    continue;
                }

                centers.Add(span.Left + (span.Width / 2));
            }

            if (centers.Count == 0)
            {
                return null;
            }

            centers.Sort();
            return centers[centers.Count / 2];
        }

        private static int EstimateShoulderY(RowSpan?[] spans, Rectangle bounds)
        {
            var from = bounds.Y + (int)Math.Round(bounds.Height * 0.13);
            var to = bounds.Y + (int)Math.Round(bounds.Height * 0.34);
            var maxWidth = 0;

            for (var y = Math.Max(0, from); y < Math.Min(spans.Length, to); y++)
            {
                if (spans[y] is RowSpan span)
                {
                    maxWidth = Math.Max(maxWidth, span.Width);
                }
            }

            var threshold = Math.Max(bounds.Width * 0.30, maxWidth * 0.72);
            for (var y = Math.Max(0, from); y < Math.Min(spans.Length, to); y++)
            {
                if (spans[y] is RowSpan span && span.Width >= threshold)
                {
                    return y;
                }
            }

            return bounds.Y + (int)Math.Round(bounds.Height * 0.23);
        }

        private static RowSpan FindBestSpanNear(RowSpan?[] spans, Rectangle bounds, int centerY, int radius)
        {
            var best = new RowSpan(bounds.X, bounds.Right);
            var bestWidth = 0;
            var from = Math.Max(bounds.Y, centerY - radius);
            var to = Math.Min(bounds.Bottom, centerY + radius + 1);

            for (var y = from; y < to && y < spans.Length; y++)
            {
                if (spans[y] is not RowSpan span)
                {
                    continue;
                }

                if (span.Width > bestWidth)
                {
                    best = span;
                    bestWidth = span.Width;
                }
            }

            return best;
        }

        private static Rectangle? DetectForegroundBounds(Bitmap image)
        {
            if (image.Width < 16 || image.Height < 16)
            {
                return null;
            }

            var background = EstimateBackgroundColor(image);
            var minX = image.Width;
            var minY = image.Height;
            var maxX = -1;
            var maxY = -1;
            const int step = 2;

            for (var y = 0; y < image.Height; y += step)
            {
                for (var x = 0; x < image.Width; x += step)
                {
                    var pixel = image.GetPixel(x, y);
                    if (ColorDistance(pixel, background) <= 42)
                    {
                        continue;
                    }

                    minX = Math.Min(minX, x);
                    minY = Math.Min(minY, y);
                    maxX = Math.Max(maxX, x);
                    maxY = Math.Max(maxY, y);
                }
            }

            if (maxX < minX || maxY < minY)
            {
                return null;
            }

            var paddingX = Math.Max(6, (int)Math.Round(image.Width * 0.025));
            var paddingY = Math.Max(6, (int)Math.Round(image.Height * 0.018));
            minX = Math.Max(0, minX - paddingX);
            minY = Math.Max(0, minY - paddingY);
            maxX = Math.Min(image.Width - 1, maxX + paddingX);
            maxY = Math.Min(image.Height - 1, maxY + paddingY);

            var width = maxX - minX + 1;
            var height = maxY - minY + 1;
            var fillsMostImage = width > image.Width * 0.92 && height > image.Height * 0.92;
            var tooSmall = width < image.Width * 0.18 || height < image.Height * 0.35;
            if (fillsMostImage || tooSmall)
            {
                return null;
            }

            return new Rectangle(minX, minY, width, height);
        }

        private static Color EstimateBackgroundColor(Bitmap image)
        {
            const int sample = 14;
            long r = 0;
            long g = 0;
            long b = 0;
            var count = 0;

            void AddSample(int startX, int startY)
            {
                var endX = Math.Min(image.Width, startX + sample);
                var endY = Math.Min(image.Height, startY + sample);
                for (var y = Math.Max(0, startY); y < endY; y++)
                {
                    for (var x = Math.Max(0, startX); x < endX; x++)
                    {
                        var pixel = image.GetPixel(x, y);
                        r += pixel.R;
                        g += pixel.G;
                        b += pixel.B;
                        count++;
                    }
                }
            }

            AddSample(0, 0);
            AddSample(Math.Max(0, image.Width - sample), 0);
            AddSample(0, Math.Max(0, image.Height - sample));
            AddSample(Math.Max(0, image.Width - sample), Math.Max(0, image.Height - sample));

            return count == 0
                ? Color.White
                : Color.FromArgb((int)(r / count), (int)(g / count), (int)(b / count));
        }

        private static double ColorDistance(Color left, Color right)
        {
            var r = left.R - right.R;
            var g = left.G - right.G;
            var b = left.B - right.B;
            return Math.Sqrt((r * r) + (g * g) + (b * b));
        }

        private static Bitmap PrepareGarmentForPlacement(Image source, string category)
        {
            return category is "galabeya" or "abaya"
                ? CreateGalabeyaLayer(source)
                : RemoveBrightBackground(source);
        }

        private static Bitmap CreateGalabeyaLayer(Image source)
        {
            var sourceRatio = source.Width / (double)Math.Max(1, source.Height);
            var width = Math.Clamp((int)Math.Round(520 * Math.Clamp(sourceRatio, 0.36, 0.58)), 210, 310);
            const int height = 720;
            var bitmap = new Bitmap(width, height, PixelFormat.Format32bppArgb);

            using var graphics = Graphics.FromImage(bitmap);
            graphics.CompositingQuality = CompositingQuality.HighQuality;
            graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
            graphics.SmoothingMode = SmoothingMode.HighQuality;
            graphics.Clear(Color.Transparent);

            var center = width / 2f;
            var shoulderY = height * 0.105f;
            var cuffY = height * 0.47f;
            var hemY = height * 0.975f;
            var shoulderHalf = width * 0.34f;
            var chestHalf = width * 0.29f;
            var waistHalf = width * 0.30f;
            var hemHalf = width * 0.43f;

            using var fabric = new LinearGradientBrush(
                new Rectangle(0, 0, width, height),
                Color.FromArgb(252, 252, 248),
                Color.FromArgb(224, 226, 221),
                LinearGradientMode.Horizontal);
            using var softShadow = new SolidBrush(Color.FromArgb(42, 36, 37, 42));
            using var seamPen = new Pen(Color.FromArgb(95, 178, 181, 176), Math.Max(1.2f, width * 0.006f));
            using var foldPen = new Pen(Color.FromArgb(62, 160, 164, 160), Math.Max(1.0f, width * 0.004f));
            using var highlightPen = new Pen(Color.FromArgb(80, 255, 255, 255), Math.Max(1.0f, width * 0.006f));

            using var robe = new GraphicsPath();
            robe.AddBezier(
                center - shoulderHalf, shoulderY,
                center - chestHalf, height * 0.24f,
                center - waistHalf, height * 0.58f,
                center - hemHalf, hemY);
            robe.AddLine(center - hemHalf, hemY, center + hemHalf, hemY);
            robe.AddBezier(
                center + hemHalf, hemY,
                center + waistHalf, height * 0.58f,
                center + chestHalf, height * 0.24f,
                center + shoulderHalf, shoulderY);
            robe.AddLine(center + shoulderHalf, shoulderY, center - shoulderHalf, shoulderY);
            robe.CloseFigure();

            using var leftSleeve = new GraphicsPath();
            leftSleeve.AddBezier(
                center - shoulderHalf + 2,
                shoulderY + 10,
                center - (width * 0.48f),
                height * 0.22f,
                center - (width * 0.49f),
                height * 0.35f,
                center - (width * 0.39f),
                cuffY);
            leftSleeve.AddLine(center - (width * 0.27f), cuffY - 6, center - chestHalf, shoulderY + 34);
            leftSleeve.CloseFigure();

            using var rightSleeve = new GraphicsPath();
            rightSleeve.AddBezier(
                center + shoulderHalf - 2,
                shoulderY + 10,
                center + (width * 0.48f),
                height * 0.22f,
                center + (width * 0.49f),
                height * 0.35f,
                center + (width * 0.39f),
                cuffY);
            rightSleeve.AddLine(center + (width * 0.27f), cuffY - 6, center + chestHalf, shoulderY + 34);
            rightSleeve.CloseFigure();

            graphics.FillPath(softShadow, robe);
            graphics.TranslateTransform(0, -4);
            graphics.FillPath(fabric, leftSleeve);
            graphics.FillPath(fabric, rightSleeve);
            graphics.FillPath(fabric, robe);
            graphics.ResetTransform();

            graphics.DrawPath(seamPen, robe);
            graphics.DrawPath(seamPen, leftSleeve);
            graphics.DrawPath(seamPen, rightSleeve);

            var neckWidth = width * 0.19f;
            var neckHeight = height * 0.038f;
            using var neckFill = new SolidBrush(Color.FromArgb(245, 246, 242));
            using var neckPen = new Pen(Color.FromArgb(130, 170, 173, 168), Math.Max(1.2f, width * 0.006f));
            graphics.FillEllipse(neckFill, center - (neckWidth / 2), shoulderY - (neckHeight / 2), neckWidth, neckHeight);
            graphics.DrawEllipse(neckPen, center - (neckWidth / 2), shoulderY - (neckHeight / 2), neckWidth, neckHeight);

            var placketTop = shoulderY + neckHeight * 0.45f;
            var placketBottom = height * 0.32f;
            graphics.DrawLine(seamPen, center, placketTop, center, placketBottom);
            graphics.DrawLine(highlightPen, center - (width * 0.032f), shoulderY + 18, center - (width * 0.10f), hemY - 18);
            graphics.DrawLine(foldPen, center + (width * 0.09f), height * 0.20f, center + (width * 0.16f), hemY - 24);
            graphics.DrawLine(foldPen, center - (width * 0.12f), height * 0.30f, center - (width * 0.18f), hemY - 28);

            using var buttonBrush = new SolidBrush(Color.FromArgb(105, 54, 54, 52));
            var buttonSize = Math.Max(3.0f, width * 0.018f);
            for (var i = 0; i < 5; i++)
            {
                var y = placketTop + 18 + (i * buttonSize * 4.2f);
                graphics.FillEllipse(buttonBrush, center - (buttonSize / 2), y, buttonSize, buttonSize);
            }

            using var hemPen = new Pen(Color.FromArgb(110, 188, 190, 184), Math.Max(1.4f, width * 0.007f));
            graphics.DrawLine(hemPen, center - hemHalf + 6, hemY - 8, center + hemHalf - 6, hemY - 8);

            return bitmap;
        }

        private static Bitmap RemoveBrightBackground(Image source)
        {
            var bitmap = new Bitmap(source.Width, source.Height, PixelFormat.Format32bppArgb);
            using (var graphics = Graphics.FromImage(bitmap))
            {
                graphics.DrawImage(source, 0, 0, source.Width, source.Height);
            }

            var visited = new bool[bitmap.Width, bitmap.Height];
            var queue = new Queue<Point>();

            void EnqueueIfBackground(int x, int y)
            {
                if (x < 0 || y < 0 || x >= bitmap.Width || y >= bitmap.Height || visited[x, y])
                {
                    return;
                }

                var pixel = bitmap.GetPixel(x, y);
                if (!IsBrightBackground(pixel))
                {
                    return;
                }

                visited[x, y] = true;
                queue.Enqueue(new Point(x, y));
            }

            for (var x = 0; x < bitmap.Width; x++)
            {
                EnqueueIfBackground(x, 0);
                EnqueueIfBackground(x, bitmap.Height - 1);
            }

            for (var y = 0; y < bitmap.Height; y++)
            {
                EnqueueIfBackground(0, y);
                EnqueueIfBackground(bitmap.Width - 1, y);
            }

            while (queue.Count > 0)
            {
                var point = queue.Dequeue();
                var pixel = bitmap.GetPixel(point.X, point.Y);
                bitmap.SetPixel(point.X, point.Y, Color.FromArgb(0, pixel.R, pixel.G, pixel.B));

                EnqueueIfBackground(point.X + 1, point.Y);
                EnqueueIfBackground(point.X - 1, point.Y);
                EnqueueIfBackground(point.X, point.Y + 1);
                EnqueueIfBackground(point.X, point.Y - 1);
            }

            var minX = bitmap.Width;
            var minY = bitmap.Height;
            var maxX = -1;
            var maxY = -1;

            for (var y = 0; y < bitmap.Height; y++)
            {
                for (var x = 0; x < bitmap.Width; x++)
                {
                    var pixel = bitmap.GetPixel(x, y);
                    if (pixel.A > 12)
                    {
                        minX = Math.Min(minX, x);
                        minY = Math.Min(minY, y);
                        maxX = Math.Max(maxX, x);
                        maxY = Math.Max(maxY, y);
                    }
                }
            }

            if (maxX < minX || maxY < minY)
            {
                return bitmap;
            }

            var padding = Math.Max(4, (int)Math.Round(Math.Max(bitmap.Width, bitmap.Height) * 0.018));
            minX = Math.Max(0, minX - padding);
            minY = Math.Max(0, minY - padding);
            maxX = Math.Min(bitmap.Width - 1, maxX + padding);
            maxY = Math.Min(bitmap.Height - 1, maxY + padding);

            var crop = Rectangle.FromLTRB(minX, minY, maxX + 1, maxY + 1);
            var cropped = new Bitmap(crop.Width, crop.Height, PixelFormat.Format32bppArgb);
            using (var graphics = Graphics.FromImage(cropped))
            {
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.DrawImage(bitmap, new Rectangle(0, 0, crop.Width, crop.Height), crop, GraphicsUnit.Pixel);
            }

            bitmap.Dispose();
            return cropped;
        }

        private static bool IsBrightBackground(Color pixel)
        {
            if (pixel.A < 8)
            {
                return true;
            }

            var brightest = Math.Max(pixel.R, Math.Max(pixel.G, pixel.B));
            var darkest = Math.Min(pixel.R, Math.Min(pixel.G, pixel.B));
            return (pixel.R > 244 && pixel.G > 244 && pixel.B > 244) ||
                (brightest > 214 && brightest - darkest < 44);
        }
    }
}
