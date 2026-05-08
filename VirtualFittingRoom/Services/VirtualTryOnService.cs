using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.Extensions.Options;
using VirtualFittingRoom.Models;

namespace VirtualFittingRoom.Services
{
    public class VirtualTryOnService
    {
        private readonly VirtualTryOnOptions _options;
        private readonly LocalInferenceServerManager _serverManager;
        private readonly IHttpClientFactory _httpClientFactory;

        public VirtualTryOnService(
            IOptions<VirtualTryOnOptions> options,
            LocalInferenceServerManager serverManager,
            IHttpClientFactory httpClientFactory)
        {
            _options = options.Value;
            _serverManager = serverManager;
            _httpClientFactory = httpClientFactory;
        }

        public async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            CancellationToken cancellationToken = default)
        {
            try
            {
                return IsApiMode()
                    ? await RunAgainstApiAsync(personImage, clothingImage, garmentArea, cancellationToken)
                    : IsReplicateMode()
                    ? await RunAgainstReplicateAsync(personImage, clothingImage, garmentArea, cancellationToken)
                    : IsHuggingFaceMode()
                    ? await RunAgainstHuggingFaceSpaceAsync(personImage, clothingImage, garmentArea, cancellationToken)
                    : await RunAgainstLocalServerAsync(personImage, clothingImage, garmentArea, cancellationToken);
            }
            catch (Exception ex)
            {
                return (false, null, $"AI inference failed: {ex.Message}");
            }
        }

        private bool IsApiMode() =>
            string.Equals(_options.Mode, "Api", StringComparison.OrdinalIgnoreCase);

        private bool IsReplicateMode() =>
            string.Equals(_options.Mode, "Replicate", StringComparison.OrdinalIgnoreCase);

        private bool IsHuggingFaceMode() =>
            string.Equals(_options.Mode, "HuggingFace", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(_options.Mode, "HuggingFaceSpace", StringComparison.OrdinalIgnoreCase);

        private async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunAgainstHuggingFaceSpaceAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            CancellationToken cancellationToken)
        {
            if (string.IsNullOrWhiteSpace(_options.HuggingFaceSpaceUrl))
            {
                return (false, null, "VirtualTryOn:HuggingFaceSpaceUrl is not configured.");
            }

            using var client = _httpClientFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(Math.Max(60, _options.ApiTimeoutSeconds));

            var hfToken = ResolveHuggingFaceToken();
            if (!string.IsNullOrWhiteSpace(hfToken))
            {
                client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", hfToken);
            }

            var baseUrl = _options.HuggingFaceSpaceUrl.TrimEnd('/');
            var apiName = _options.HuggingFaceApiName.Trim().Trim('/');

            var personFile = await UploadGradioFileAsync(client, baseUrl, personImage, "person.png", cancellationToken);
            var garmentFile = await UploadGradioFileAsync(client, baseUrl, clothingImage, "garment.png", cancellationToken);

            var request = new
            {
                data = new object?[]
                {
                    new
                    {
                        background = personFile,
                        layers = Array.Empty<object>(),
                        composite = (object?)null
                    },
                    garmentFile,
                    BuildGarmentDescription(garmentArea),
                    _options.HuggingFaceAutoMask,
                    _options.HuggingFaceAutoCrop,
                    Math.Clamp(_options.HuggingFaceDenoiseSteps, 10, 50),
                    _options.HuggingFaceSeed
                }
            };

            using var submitResponse = await client.PostAsJsonAsync(
                $"{baseUrl}/call/{apiName}",
                request,
                cancellationToken);

            var submitJson = await submitResponse.Content.ReadAsStringAsync(cancellationToken);
            if (!submitResponse.IsSuccessStatusCode)
            {
                return (false, null, $"Hugging Face Space returned HTTP {(int)submitResponse.StatusCode}: {submitJson}");
            }

            var eventId = ReadJsonString(submitJson, "event_id");
            if (string.IsNullOrWhiteSpace(eventId))
            {
                return (false, null, "Hugging Face Space did not return a queue event id.");
            }

            using var resultResponse = await client.GetAsync(
                $"{baseUrl}/call/{apiName}/{Uri.EscapeDataString(eventId)}",
                cancellationToken);

            var eventStream = await resultResponse.Content.ReadAsStringAsync(cancellationToken);
            if (!resultResponse.IsSuccessStatusCode)
            {
                return (false, null, $"Hugging Face queue returned HTTP {(int)resultResponse.StatusCode}: {eventStream}");
            }

            var outputUrl = ExtractFirstGradioOutputUrl(eventStream, baseUrl);
            if (string.IsNullOrWhiteSpace(outputUrl))
            {
                return (false, null, BuildHuggingFaceNoOutputError(eventStream));
            }

            return (true, await client.GetByteArrayAsync(outputUrl, cancellationToken), null);
        }

        private async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunAgainstReplicateAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            CancellationToken cancellationToken)
        {
            var apiToken = ResolveReplicateApiToken();
            if (string.IsNullOrWhiteSpace(apiToken))
            {
                return (false, null, "Replicate API token is not configured. Set VirtualTryOn:ReplicateApiToken or REPLICATE_API_TOKEN.");
            }

            if (string.IsNullOrWhiteSpace(_options.ReplicateVersion))
            {
                return (false, null, "VirtualTryOn:ReplicateVersion is not configured.");
            }

            using var client = _httpClientFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(Math.Max(30, _options.ApiTimeoutSeconds));
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiToken);
            client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
            client.DefaultRequestHeaders.TryAddWithoutValidation(
                "Prefer",
                $"wait={Math.Clamp(_options.ReplicateWaitSeconds, 1, 60)}");

            var input = new Dictionary<string, object?>
            {
                [_options.ReplicatePersonFieldName] = ToDataUrl(personImage),
                [_options.ReplicateClothingFieldName] = ToDataUrl(clothingImage),
                [_options.ReplicateCategoryFieldName] = NormalizeGarmentArea(garmentArea),
                ["output_format"] = string.IsNullOrWhiteSpace(_options.ReplicateOutputFormat)
                    ? "png"
                    : _options.ReplicateOutputFormat
            };

            var createRequest = new
            {
                version = _options.ReplicateVersion,
                input
            };

            using var createResponse = await client.PostAsJsonAsync(
                "https://api.replicate.com/v1/predictions",
                createRequest,
                cancellationToken);

            var predictionJson = await createResponse.Content.ReadAsStringAsync(cancellationToken);
            if (!createResponse.IsSuccessStatusCode)
            {
                return (false, null, $"Replicate returned HTTP {(int)createResponse.StatusCode}: {predictionJson}");
            }

            var prediction = ReplicatePrediction.FromJson(predictionJson);
            var deadline = DateTimeOffset.UtcNow.AddSeconds(Math.Max(60, _options.ApiTimeoutSeconds));

            while (!prediction.IsTerminal && DateTimeOffset.UtcNow < deadline)
            {
                if (string.IsNullOrWhiteSpace(prediction.GetUrl))
                {
                    return (false, null, "Replicate prediction is still running but did not return a polling URL.");
                }

                await Task.Delay(TimeSpan.FromSeconds(Math.Max(1, _options.ReplicatePollSeconds)), cancellationToken);

                using var pollResponse = await client.GetAsync(prediction.GetUrl, cancellationToken);
                var pollJson = await pollResponse.Content.ReadAsStringAsync(cancellationToken);
                if (!pollResponse.IsSuccessStatusCode)
                {
                    return (false, null, $"Replicate polling returned HTTP {(int)pollResponse.StatusCode}: {pollJson}");
                }

                prediction = ReplicatePrediction.FromJson(pollJson);
            }

            if (!prediction.IsTerminal)
            {
                return (false, null, "Replicate prediction timed out before completion.");
            }

            if (!string.Equals(prediction.Status, "succeeded", StringComparison.OrdinalIgnoreCase))
            {
                return (false, null, prediction.Error ?? $"Replicate prediction ended with status '{prediction.Status}'.");
            }

            var output = prediction.FirstOutput;
            if (string.IsNullOrWhiteSpace(output))
            {
                return (false, null, "Replicate completed but did not return an output image.");
            }

            return await ResolveReplicateOutputAsync(client, output, cancellationToken);
        }

        private async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunAgainstApiAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            CancellationToken cancellationToken)
        {
            if (string.IsNullOrWhiteSpace(_options.ApiUrl))
            {
                return (false, null, "VirtualTryOn:ApiUrl is not configured yet in appsettings.json.");
            }

            using var client = _httpClientFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(Math.Max(30, _options.ApiTimeoutSeconds));

            using var content = new MultipartFormDataContent();
            content.Add(CreateImageContent(personImage, "person.png"), _options.ApiPersonFieldName, "person.png");
            content.Add(CreateImageContent(clothingImage, "cloth.png"), _options.ApiClothingFieldName, "cloth.png");
            content.Add(new StringContent(garmentArea), _options.ApiCategoryFieldName);

            if (!string.IsNullOrWhiteSpace(_options.ApiKey))
            {
                if (string.Equals(_options.ApiKeyHeader, "Authorization", StringComparison.OrdinalIgnoreCase))
                {
                    client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _options.ApiKey);
                }
                else
                {
                    client.DefaultRequestHeaders.Remove(_options.ApiKeyHeader);
                    client.DefaultRequestHeaders.Add(_options.ApiKeyHeader, _options.ApiKey);
                }
            }

            using var response = await client.PostAsync(_options.ApiUrl, content, cancellationToken);
            var responseBytes = await response.Content.ReadAsByteArrayAsync(cancellationToken);

            if (!response.IsSuccessStatusCode)
            {
                var responseText = SafeReadText(responseBytes);
                return (false, null, $"API returned HTTP {(int)response.StatusCode}: {responseText}");
            }

            if (IsImageResponse(response.Content.Headers.ContentType?.MediaType))
            {
                return (true, responseBytes, null);
            }

            var responseTextJson = SafeReadText(responseBytes);
            if (string.IsNullOrWhiteSpace(responseTextJson))
            {
                return (false, null, "API returned an empty response.");
            }

            using var document = JsonDocument.Parse(responseTextJson);
            if (document.RootElement.TryGetProperty("error", out var errorNode))
            {
                return (false, null, errorNode.GetString() ?? "API returned an error.");
            }

            if (!document.RootElement.TryGetProperty(_options.ApiResponseImageField, out var outputNode))
            {
                return (false, null, $"API response does not contain '{_options.ApiResponseImageField}'.");
            }

            var outputValue = outputNode.GetString();
            if (string.IsNullOrWhiteSpace(outputValue))
            {
                return (false, null, "API response image field was empty.");
            }

            if (outputValue.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
            {
                var commaIndex = outputValue.IndexOf(',');
                if (commaIndex >= 0 && commaIndex < outputValue.Length - 1)
                {
                    outputValue = outputValue[(commaIndex + 1)..];
                }
            }

            return (true, Convert.FromBase64String(outputValue), null);
        }

        private async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunAgainstLocalServerAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            CancellationToken cancellationToken)
        {
            var serverState = await _serverManager.EnsureServerReadyAsync(cancellationToken);
            if (!serverState.Success)
            {
                return (false, null, serverState.Error);
            }

            using var client = _httpClientFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(Math.Max(30, _options.ApiTimeoutSeconds));

            var request = new
            {
                personImageBase64 = Convert.ToBase64String(personImage),
                clothingImageBase64 = Convert.ToBase64String(clothingImage),
                category = garmentArea
            };

            using var response = await client.PostAsJsonAsync(
                $"{_serverManager.ServerUrl}/tryon",
                request,
                cancellationToken);

            var json = await response.Content.ReadAsStringAsync(cancellationToken);
            using var document = JsonDocument.Parse(json);

            if (!response.IsSuccessStatusCode)
            {
                var error = document.RootElement.TryGetProperty("error", out var errorNode)
                    ? errorNode.GetString()
                    : $"Local AI server returned HTTP {(int)response.StatusCode}.";

                return (false, null, $"AI inference failed: {error}");
            }

            if (!document.RootElement.TryGetProperty("outputImageBase64", out var outputNode))
            {
                return (false, null, "AI inference completed but no output image was returned.");
            }

            var outputBase64 = outputNode.GetString();
            if (string.IsNullOrWhiteSpace(outputBase64))
            {
                return (false, null, "AI inference completed but returned an empty image.");
            }

            return (true, Convert.FromBase64String(outputBase64), null);
        }

        private static ByteArrayContent CreateImageContent(byte[] bytes, string fileName)
        {
            var content = new ByteArrayContent(bytes);
            content.Headers.ContentType = new MediaTypeHeaderValue("image/png");
            content.Headers.ContentDisposition = new ContentDispositionHeaderValue("form-data")
            {
                FileName = fileName,
                Name = fileName
            };
            return content;
        }

        private static bool IsImageResponse(string? mediaType) =>
            !string.IsNullOrWhiteSpace(mediaType) &&
            mediaType.StartsWith("image/", StringComparison.OrdinalIgnoreCase);

        private string ResolveReplicateApiToken()
        {
            if (!string.IsNullOrWhiteSpace(_options.ReplicateApiToken) &&
                !_options.ReplicateApiToken.StartsWith("PUT_", StringComparison.OrdinalIgnoreCase))
            {
                return _options.ReplicateApiToken;
            }

            return Environment.GetEnvironmentVariable("REPLICATE_API_TOKEN") ?? string.Empty;
        }

        private string ResolveHuggingFaceToken()
        {
            if (!string.IsNullOrWhiteSpace(_options.HuggingFaceToken) &&
                !_options.HuggingFaceToken.StartsWith("PUT_", StringComparison.OrdinalIgnoreCase))
            {
                return _options.HuggingFaceToken;
            }

            return Environment.GetEnvironmentVariable("HF_TOKEN") ??
                   Environment.GetEnvironmentVariable("HUGGINGFACE_TOKEN") ??
                   string.Empty;
        }

        private string BuildGarmentDescription(string garmentArea)
        {
            if (!string.IsNullOrWhiteSpace(_options.HuggingFaceGarmentDescription))
            {
                return _options.HuggingFaceGarmentDescription;
            }

            return NormalizeGarmentArea(garmentArea) switch
            {
                "lower" => "Lower body clothing garment",
                "overall" => "Full body clothing garment",
                _ => "Upper body clothing garment"
            };
        }

        private static async Task<object> UploadGradioFileAsync(
            HttpClient client,
            string baseUrl,
            byte[] imageBytes,
            string fileName,
            CancellationToken cancellationToken)
        {
            using var content = new MultipartFormDataContent();
            content.Add(CreateGradioUploadContent(imageBytes), "files", fileName);

            using var response = await client.PostAsync($"{baseUrl}/upload", content, cancellationToken);
            var json = await response.Content.ReadAsStringAsync(cancellationToken);
            if (!response.IsSuccessStatusCode)
            {
                throw new InvalidOperationException($"Hugging Face upload returned HTTP {(int)response.StatusCode}: {json}");
            }

            using var document = JsonDocument.Parse(json);
            if (document.RootElement.ValueKind == JsonValueKind.String)
            {
                var uploadedPath = document.RootElement.GetString();
                if (string.IsNullOrWhiteSpace(uploadedPath))
                {
                    throw new InvalidOperationException("Hugging Face upload returned an empty file path.");
                }

                return CreateGradioFileData(uploadedPath, baseUrl, fileName, imageBytes.LongLength);
            }

            var file = document.RootElement;
            if (document.RootElement.ValueKind == JsonValueKind.Array)
            {
                var uploadedFiles = document.RootElement.EnumerateArray().ToList();
                if (uploadedFiles.Count == 0)
                {
                    throw new InvalidOperationException($"Hugging Face upload returned no files. Raw response: {json}");
                }

                file = uploadedFiles[0];
                if (file.ValueKind == JsonValueKind.String)
                {
                    var uploadedPath = file.GetString();
                    if (string.IsNullOrWhiteSpace(uploadedPath))
                    {
                        throw new InvalidOperationException("Hugging Face upload returned an empty file path.");
                    }

                    return CreateGradioFileData(uploadedPath, baseUrl, fileName, imageBytes.LongLength);
                }
            }

            var path = TryGetJsonString(file, "path") ?? TryGetJsonString(file, "name");
            if (string.IsNullOrWhiteSpace(path))
            {
                throw new InvalidOperationException("Hugging Face upload did not return a file path.");
            }

            return CreateGradioFileData(
                path,
                baseUrl,
                TryGetJsonString(file, "orig_name") ?? fileName,
                TryGetJsonInt64(file, "size") ?? imageBytes.LongLength,
                TryGetJsonString(file, "url"),
                TryGetJsonString(file, "mime_type") ?? "image/png");
        }

        private static Dictionary<string, object?> CreateGradioFileData(
            string path,
            string baseUrl,
            string fileName,
            long size,
            string? url = null,
            string mimeType = "image/png")
        {
            return new Dictionary<string, object?>
            {
                ["path"] = path,
                ["url"] = string.IsNullOrWhiteSpace(url) ? $"{baseUrl}/file={Uri.EscapeDataString(path)}" : url,
                ["orig_name"] = fileName,
                ["size"] = size,
                ["mime_type"] = mimeType,
                ["meta"] = new Dictionary<string, object?>
                {
                    ["_type"] = "gradio.FileData"
                }
            };
        }

        private static ByteArrayContent CreateGradioUploadContent(byte[] bytes)
        {
            var content = new ByteArrayContent(bytes);
            content.Headers.ContentType = new MediaTypeHeaderValue("image/png");
            return content;
        }

        private static string? ExtractFirstGradioOutputUrl(string eventStream, string baseUrl)
        {
            var dataLines = eventStream
                .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Where(line => line.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
                .Select(line => line["data:".Length..].Trim())
                .Where(data => !string.IsNullOrWhiteSpace(data))
                .Reverse();

            foreach (var dataJson in dataLines)
            {
                try
                {
                    using var document = JsonDocument.Parse(dataJson);
                    var outputUrl = FindFirstFileUrl(document.RootElement, baseUrl);
                    if (!string.IsNullOrWhiteSpace(outputUrl))
                    {
                        return outputUrl;
                    }
                }
                catch (JsonException)
                {
                    continue;
                }
            }

            return null;
        }

        private static string BuildHuggingFaceNoOutputError(string eventStream)
        {
            var raw = TrimForError(eventStream);
            if (eventStream.Contains("event: error", StringComparison.OrdinalIgnoreCase))
            {
                return "Hugging Face Space failed while running the model. This public free Space often fails when ZeroGPU quota is exhausted, the queue is overloaded, or anonymous calls are blocked. Add a free Hugging Face token as HF_TOKEN, then try again. Raw response: " + raw;
            }

            return "Hugging Face Space completed but did not return an output image. Raw response: " + raw;
        }

        private static string? FindFirstFileUrl(JsonElement element, string baseUrl)
        {
            if (element.ValueKind == JsonValueKind.Object)
            {
                var url = TryGetJsonString(element, "url");
                if (!string.IsNullOrWhiteSpace(url))
                {
                    return url;
                }

                var path = TryGetJsonString(element, "path");
                if (!string.IsNullOrWhiteSpace(path))
                {
                    return path.StartsWith("http", StringComparison.OrdinalIgnoreCase)
                        ? path
                        : $"{baseUrl}/file={Uri.EscapeDataString(path)}";
                }

                foreach (var property in element.EnumerateObject())
                {
                    var nested = FindFirstFileUrl(property.Value, baseUrl);
                    if (!string.IsNullOrWhiteSpace(nested))
                    {
                        return nested;
                    }
                }
            }

            if (element.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in element.EnumerateArray())
                {
                    var nested = FindFirstFileUrl(item, baseUrl);
                    if (!string.IsNullOrWhiteSpace(nested))
                    {
                        return nested;
                    }
                }
            }

            return null;
        }

        private static string? ReadJsonString(string json, string propertyName)
        {
            using var document = JsonDocument.Parse(json);
            return TryGetJsonString(document.RootElement, propertyName);
        }

        private static string? TryGetJsonString(JsonElement element, string propertyName)
        {
            if (!element.TryGetProperty(propertyName, out var property) ||
                property.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
            {
                return null;
            }

            return property.ValueKind == JsonValueKind.String
                ? property.GetString()
                : property.ToString();
        }

        private static long? TryGetJsonInt64(JsonElement element, string propertyName)
        {
            return element.TryGetProperty(propertyName, out var property) &&
                   property.TryGetInt64(out var value)
                ? value
                : null;
        }

        private static string TrimForError(string value)
        {
            const int maxLength = 1200;
            if (string.IsNullOrWhiteSpace(value))
            {
                return "(empty)";
            }

            var compact = value.Replace("\r", string.Empty).Replace("\n", " ");
            return compact.Length <= maxLength ? compact : compact[..maxLength] + "...";
        }

        private static string ToDataUrl(byte[] imageBytes) =>
            $"data:image/png;base64,{Convert.ToBase64String(imageBytes)}";

        private static string NormalizeGarmentArea(string garmentArea)
        {
            return garmentArea.Trim().ToLowerInvariant() switch
            {
                "upper_body" => "upper",
                "lower_body" => "lower",
                "dress" => "overall",
                "dresses" => "overall",
                "overall" => "overall",
                "lower" => "lower",
                _ => "upper"
            };
        }

        private static async Task<(bool Success, byte[]? OutputBytes, string? Error)> ResolveReplicateOutputAsync(
            HttpClient client,
            string output,
            CancellationToken cancellationToken)
        {
            if (output.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
            {
                var commaIndex = output.IndexOf(',');
                if (commaIndex >= 0 && commaIndex < output.Length - 1)
                {
                    return (true, Convert.FromBase64String(output[(commaIndex + 1)..]), null);
                }
            }

            if (Uri.TryCreate(output, UriKind.Absolute, out var outputUri))
            {
                return (true, await client.GetByteArrayAsync(outputUri, cancellationToken), null);
            }

            try
            {
                return (true, Convert.FromBase64String(output), null);
            }
            catch
            {
                return (false, null, "Replicate returned an output format that could not be read as an image.");
            }
        }

        private static string SafeReadText(byte[] responseBytes)
        {
            try
            {
                return System.Text.Encoding.UTF8.GetString(responseBytes);
            }
            catch
            {
                return "Unable to decode API response.";
            }
        }

        private sealed class ReplicatePrediction
        {
            public string? Status { get; private init; }
            public string? Error { get; private init; }
            public string? GetUrl { get; private init; }
            public string? FirstOutput { get; private init; }

            public bool IsTerminal =>
                string.Equals(Status, "succeeded", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(Status, "successful", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(Status, "failed", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(Status, "canceled", StringComparison.OrdinalIgnoreCase);

            public static ReplicatePrediction FromJson(string json)
            {
                using var document = JsonDocument.Parse(json);
                var root = document.RootElement;

                return new ReplicatePrediction
                {
                    Status = TryGetString(root, "status"),
                    Error = TryGetString(root, "error"),
                    GetUrl = TryGetNestedString(root, "urls", "get"),
                    FirstOutput = ReadOutput(root)
                };
            }

            private static string? ReadOutput(JsonElement root)
            {
                if (!root.TryGetProperty("output", out var output) ||
                    output.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
                {
                    return null;
                }

                if (output.ValueKind == JsonValueKind.String)
                {
                    return output.GetString();
                }

                if (output.ValueKind == JsonValueKind.Array)
                {
                    foreach (var item in output.EnumerateArray())
                    {
                        if (item.ValueKind == JsonValueKind.String)
                        {
                            return item.GetString();
                        }
                    }
                }

                return null;
            }

            private static string? TryGetString(JsonElement element, string propertyName)
            {
                if (!element.TryGetProperty(propertyName, out var property) ||
                    property.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
                {
                    return null;
                }

                return property.ValueKind == JsonValueKind.String
                    ? property.GetString()
                    : property.ToString();
            }

            private static string? TryGetNestedString(JsonElement element, string parentName, string propertyName)
            {
                return element.TryGetProperty(parentName, out var parent)
                    ? TryGetString(parent, propertyName)
                    : null;
            }
        }
    }
}
