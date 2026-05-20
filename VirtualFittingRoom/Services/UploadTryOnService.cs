using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Collections.Generic;
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
            return await RunAsync(personImage, clothingImage, garmentArea, clothingType, null, cancellationToken);
        }

        public async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            string? clothingType,
            string? apiUrlOverride,
            CancellationToken cancellationToken = default)
        {
            var aiResult = !string.IsNullOrWhiteSpace(apiUrlOverride)
                ? await _virtualTryOnService.RunApiAsync(
                    personImage,
                    clothingImage,
                    garmentArea,
                    apiUrlOverride,
                    cancellationToken)
                : await _virtualTryOnService.RunAsync(
                    personImage,
                    clothingImage,
                    garmentArea,
                    cancellationToken);

            if (aiResult.Success)
            {
                return aiResult;
            }

            if (!ShouldUsePreviewFallback())
            {
                return aiResult;
            }

            try
            {
                return (true, ComposeTryOnPreview(personImage, clothingImage, garmentArea, clothingType), null);
            }
            catch (Exception ex)
            {
                return (false, null, $"Upload try-on failed: {ex.Message}");
            }
        }

        private bool HasConfiguredCustomSpace()
        {
            var url = _options.HuggingFaceSpaceUrl?.Trim() ?? string.Empty;
            return url.Length > 0 &&
                !url.Contains("yisol-idm-vton", StringComparison.OrdinalIgnoreCase) &&
                !url.Contains("yisol/idm-vton", StringComparison.OrdinalIgnoreCase);
        }

        private bool ShouldUsePreviewFallback()
        {
            var mode = (_options.Mode ?? string.Empty).Trim();
            return string.Equals(mode, "Preview", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(mode, "Fallback", StringComparison.OrdinalIgnoreCase);
        }

        private static byte[] ComposeTryOnPreview(byte[] personBytes, byte[] garmentBytes, string garmentArea, string? clothingType)
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
                var personBounds = GetPersonReferenceBounds(person);
                var target = BuildTargetRectangle(personBounds, placement);
                using var preparedGarment = PrepareGarmentForPlacement(garment, category);
                var drawTarget = FitGarment(preparedGarment.Size, target, placement);
                graphics.DrawImage(preparedGarment, drawTarget);
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

        private static Rectangle BuildTargetRectangle(Rectangle reference, GarmentPlacement placement)
        {
            return new Rectangle(
                reference.X + (int)Math.Round(reference.Width * placement.X),
                reference.Y + (int)Math.Round(reference.Height * placement.Y),
                (int)Math.Round(reference.Width * placement.Width),
                (int)Math.Round(reference.Height * placement.Height));
        }

        private static GarmentPlacement ResolvePlacement(string garmentArea, string? clothingType)
        {
            var category = NormalizeCategory(clothingType);
            var area = (garmentArea ?? "upper").Trim().ToLowerInvariant();

            return category switch
            {
                "hoodie" => new GarmentPlacement("upper", 0.08, 0.20, 0.84, 0.46, 1.00, 0.00),
                "jacket" => new GarmentPlacement("upper", 0.08, 0.20, 0.84, 0.46, 1.00, 0.00),
                "shirt" => new GarmentPlacement("upper", 0.12, 0.22, 0.76, 0.40, 1.00, 0.00),
                "t-shirt" => new GarmentPlacement("upper", 0.11, 0.22, 0.78, 0.40, 1.00, 0.00),
                "pants" => new GarmentPlacement("lower", 0.24, 0.45, 0.52, 0.52, 1.00, 0.00),
                "shorts" => new GarmentPlacement("lower", 0.26, 0.45, 0.48, 0.34, 1.00, 0.00),
                "dress" => new GarmentPlacement("overall", 0.17, 0.15, 0.66, 0.76, 1.00, 0.00),
                "galabeya" => new GarmentPlacement("overall", 0.04, 0.19, 0.92, 0.78, 1.00, 0.00),
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
                "hoodies" => "hoodie",
                "coats" => "jacket",
                "trouser" => "pants",
                "trousers" => "pants",
                "jeans" => "pants",
                "short" => "shorts",
                "galabiya" => "galabeya",
                "jellabiya" => "galabeya",
                "jalabiya" => "galabeya",
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
                bounds.Y + ((bounds.Height - height) / 2) + (int)Math.Round(bounds.Height * placement.YBias),
                width,
                height);
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
            return category == "galabeya"
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
