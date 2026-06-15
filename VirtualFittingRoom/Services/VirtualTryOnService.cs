using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Drawing;
using System.Drawing.Imaging;
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
            return await RunAsync(personImage, clothingImage, garmentArea, null, null, cancellationToken);
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
            string? garmentView,
            CancellationToken cancellationToken = default)
        {
            return await RunAsync(personImage, clothingImage, garmentArea, clothingType, garmentView, null, cancellationToken);
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
            try
            {
                var normalizedGarmentArea = NormalizeGarmentArea(garmentArea);
                var normalizedGarmentView = NormalizeGarmentView(garmentView);
                var localMode = IsLocalMode();
                var huggingFaceMode = IsHuggingFaceMode();
                personImage = ResizeImageForInference(personImage, localMode ? 560 : huggingFaceMode ? 720 : 900, 76);
                clothingImage = ResizeImageForInference(clothingImage, localMode ? 420 : huggingFaceMode ? 560 : 700, 76);

                return IsApiMode()
                    ? await RunAgainstApiAsync(personImage, clothingImage, normalizedGarmentArea, cancellationToken, null, clothingType, normalizedGarmentView, poseLandmarksData)
                    : IsReplicateMode()
                    ? await RunAgainstReplicateAsync(personImage, clothingImage, normalizedGarmentArea, cancellationToken)
                    : IsHuggingFaceMode()
                    ? await RunAgainstHuggingFaceSpaceAsync(personImage, clothingImage, normalizedGarmentArea, clothingType, normalizedGarmentView, poseLandmarksData, cancellationToken)
                    : await RunAgainstLocalServerAsync(personImage, clothingImage, normalizedGarmentArea, clothingType, normalizedGarmentView, cancellationToken);
            }
            catch (Exception ex)
            {
                return (false, null, BuildInferenceExceptionMessage(ex));
            }
        }

        public async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunHuggingFaceAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            CancellationToken cancellationToken = default)
        {
            try
            {
                var normalizedGarmentArea = NormalizeGarmentArea(garmentArea);
                personImage = ResizeImageForInference(personImage, 720, 76);
                clothingImage = ResizeImageForInference(clothingImage, 560, 76);

                return await RunAgainstHuggingFaceSpaceAsync(
                    personImage,
                    clothingImage,
                    normalizedGarmentArea,
                    null,
                    null,
                    null,
                    cancellationToken);
            }
            catch (Exception ex)
            {
                return (false, null, BuildInferenceExceptionMessage(ex));
            }
        }

        public async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunApiAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            string apiUrl,
            CancellationToken cancellationToken = default)
        {
            return await RunApiAsync(personImage, clothingImage, garmentArea, null, null, apiUrl, cancellationToken);
        }

        public async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunApiAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            string? clothingType,
            string? garmentView,
            string apiUrl,
            CancellationToken cancellationToken = default)
        {
            return await RunApiAsync(personImage, clothingImage, garmentArea, clothingType, garmentView, apiUrl, null, cancellationToken);
        }

        public async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunApiAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            string? clothingType,
            string? garmentView,
            string apiUrl,
            string? poseLandmarksData,
            CancellationToken cancellationToken = default)
        {
            try
            {
                var normalizedGarmentArea = NormalizeGarmentArea(garmentArea);
                var normalizedGarmentView = NormalizeGarmentView(garmentView);
                personImage = ResizeImageForInference(personImage, 900, 84);
                clothingImage = ResizeImageForInference(clothingImage, 700, 84);
                return await RunAgainstApiAsync(personImage, clothingImage, normalizedGarmentArea, cancellationToken, apiUrl, clothingType, normalizedGarmentView, poseLandmarksData);
            }
            catch (Exception ex)
            {
                return (false, null, BuildInferenceExceptionMessage(ex));
            }
        }

        private bool IsApiMode() =>
            string.Equals(_options.Mode, "Api", StringComparison.OrdinalIgnoreCase);

        private bool IsReplicateMode() =>
            string.Equals(_options.Mode, "Replicate", StringComparison.OrdinalIgnoreCase);

        private bool IsHuggingFaceMode() =>
            string.Equals(_options.Mode, "HuggingFace", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(_options.Mode, "HuggingFaceSpace", StringComparison.OrdinalIgnoreCase);

        private bool IsLocalMode() =>
            !IsApiMode() &&
            !IsReplicateMode() &&
            !IsHuggingFaceMode();

        private async Task<(bool Success, byte[]? OutputBytes, string? Error)> RunAgainstHuggingFaceSpaceAsync(
            byte[] personImage,
            byte[] clothingImage,
            string garmentArea,
            string? clothingType,
            string? garmentView,
            string? poseLandmarksData,
            CancellationToken cancellationToken)
        {
            if (string.IsNullOrWhiteSpace(_options.HuggingFaceSpaceUrl))
            {
                return (false, null, "VirtualTryOn:HuggingFaceSpaceUrl is not configured.");
            }

            if (_options.HuggingFaceSpaceUrl.Contains("YOUR_USERNAME", StringComparison.OrdinalIgnoreCase) ||
                _options.HuggingFaceSpaceUrl.Contains("YOUR_SPACE_NAME", StringComparison.OrdinalIgnoreCase) ||
                _options.HuggingFaceSpaceUrl.Contains("yisol-idm-vton", StringComparison.OrdinalIgnoreCase))
            {
                return (false, null, "Hugging Face Space URL is not configured. Create your own Space from VirtualFittingRoom/hf_space, then set VirtualTryOn:HuggingFaceSpaceUrl to its stable .hf.space URL.");
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

            var personUpload = await UploadGradioFileAsync(client, baseUrl, personImage, "person.png", null, cancellationToken);
            var garmentUpload = await UploadGradioFileAsync(client, baseUrl, clothingImage, "garment.png", personUpload.ApiPrefix, cancellationToken);
            var personFile = personUpload.FileData;
            var garmentFile = garmentUpload.FileData;
            var gradioApiPrefix = garmentUpload.ApiPrefix;
            var shouldSendPose = !string.IsNullOrWhiteSpace(poseLandmarksData);

            object BuildRequest(bool includePoseLandmarks)
            {
                var data = new List<object?>
                {
                    new
                    {
                        background = personFile,
                        layers = Array.Empty<object>(),
                        composite = (object?)null
                    },
                    garmentFile,
                    BuildGarmentDescription(garmentArea, clothingType, garmentView),
                    _options.HuggingFaceAutoMask,
                    _options.HuggingFaceAutoCrop,
                    Math.Clamp(_options.HuggingFaceDenoiseSteps, 8, 24),
                    _options.HuggingFaceSeed
                };

                if (includePoseLandmarks)
                {
                    data.Add(poseLandmarksData ?? string.Empty);
                }

                return new { data = data.ToArray() };
            }

            using var submitResponse = await client.PostAsJsonAsync(
                $"{baseUrl}{gradioApiPrefix}/call/{apiName}",
                BuildRequest(shouldSendPose),
                cancellationToken);

            var submitJson = await submitResponse.Content.ReadAsStringAsync(cancellationToken);
            if (!submitResponse.IsSuccessStatusCode)
            {
                if (shouldSendPose && IsGradioInputCountError(submitJson))
                {
                    using var retryResponse = await client.PostAsJsonAsync(
                        $"{baseUrl}{gradioApiPrefix}/call/{apiName}",
                        BuildRequest(false),
                        cancellationToken);

                    var retryJson = await retryResponse.Content.ReadAsStringAsync(cancellationToken);
                    if (!retryResponse.IsSuccessStatusCode)
                    {
                        return (false, null, BuildHuggingFaceHttpError("submit the try-on request", (int)retryResponse.StatusCode, retryJson, baseUrl));
                    }

                    return await ReadHuggingFaceQueuedResultAsync(client, baseUrl, gradioApiPrefix, apiName, retryJson, cancellationToken);
                }

                return (false, null, BuildHuggingFaceHttpError("submit the try-on request", (int)submitResponse.StatusCode, submitJson, baseUrl));
            }

            return await ReadHuggingFaceQueuedResultAsync(client, baseUrl, gradioApiPrefix, apiName, submitJson, cancellationToken);
        }

        private static bool IsGradioInputCountError(string responseBody)
        {
            return responseBody.Contains("Expected", StringComparison.OrdinalIgnoreCase) &&
                   responseBody.Contains("arguments", StringComparison.OrdinalIgnoreCase);
        }

        private async Task<(bool Success, byte[]? OutputBytes, string? Error)> ReadHuggingFaceQueuedResultAsync(
            HttpClient client,
            string baseUrl,
            string gradioApiPrefix,
            string apiName,
            string submitJson,
            CancellationToken cancellationToken)
        {
            var eventId = ReadJsonString(submitJson, "event_id");
            if (string.IsNullOrWhiteSpace(eventId))
            {
                return (false, null, "Hugging Face Space did not return a queue event id.");
            }

            using var resultResponse = await client.GetAsync(
                $"{baseUrl}{gradioApiPrefix}/call/{apiName}/{Uri.EscapeDataString(eventId)}",
                cancellationToken);

            var eventStream = await resultResponse.Content.ReadAsStringAsync(cancellationToken);
            if (!resultResponse.IsSuccessStatusCode)
            {
                return (false, null, BuildHuggingFaceHttpError("read the try-on result", (int)resultResponse.StatusCode, eventStream, baseUrl));
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
            CancellationToken cancellationToken,
            string? apiUrlOverride = null,
            string? clothingType = null,
            string? garmentView = null,
            string? poseLandmarksData = null)
        {
            var apiUrl = string.IsNullOrWhiteSpace(apiUrlOverride)
                ? _options.ApiUrl?.Trim()
                : apiUrlOverride.Trim();

            if (string.IsNullOrWhiteSpace(apiUrl))
            {
                return (false, null, "Colab API URL is missing. Paste the URL ending with /tryon in the Upload page.");
            }

            if (apiUrl.Contains("PUT-YOUR-COLAB-TUNNEL-HERE", StringComparison.OrdinalIgnoreCase) ||
                apiUrl.Contains("PUT_YOUR_TRYON_API_URL_HERE", StringComparison.OrdinalIgnoreCase) ||
                apiUrl.Contains("PASTE_", StringComparison.OrdinalIgnoreCase))
            {
                return (false, null, "Colab API URL is still the placeholder. Run the Colab notebook, copy the printed URL ending with /tryon, and paste it in the Upload page.");
            }

            using var client = _httpClientFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(Math.Max(30, _options.ApiTimeoutSeconds));

            using var content = new MultipartFormDataContent();
            content.Add(CreateImageContent(personImage, "person.png"), _options.ApiPersonFieldName, "person.png");
            content.Add(CreateImageContent(clothingImage, "cloth.png"), _options.ApiClothingFieldName, "cloth.png");
            content.Add(new StringContent(garmentArea), _options.ApiCategoryFieldName);
            content.Add(new StringContent(NormalizeClothingType(clothingType)), "clothing_type");
            content.Add(new StringContent(NormalizeGarmentView(garmentView)), "garment_view");
            if (!string.IsNullOrWhiteSpace(poseLandmarksData))
            {
                content.Add(new StringContent(poseLandmarksData), "pose_landmarks_data");
            }

            if (!string.IsNullOrWhiteSpace(_options.ApiKey) &&
                !_options.ApiKey.StartsWith("PUT_", StringComparison.OrdinalIgnoreCase))
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

            using var response = await client.PostAsync(apiUrl, content, cancellationToken);
            var responseBytes = await response.Content.ReadAsByteArrayAsync(cancellationToken);

            if (!response.IsSuccessStatusCode)
            {
                var responseText = SafeReadText(responseBytes);
                return (false, null, BuildApiErrorMessage(apiUrl, (int)response.StatusCode, responseText));
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
            string? clothingType,
            string? garmentView,
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
                category = garmentArea,
                clothingType = NormalizeClothingType(clothingType),
                garmentView = NormalizeGarmentView(garmentView)
            };

            using var response = await PostLocalTryOnAsync(client, request, cancellationToken);
            var json = await response.Content.ReadAsStringAsync(cancellationToken);
            if (string.IsNullOrWhiteSpace(json))
            {
                return (false, null, $"AI inference failed: Local AI server returned an empty response from {_serverManager.ServerUrl}/tryon.");
            }

            JsonDocument document;
            try
            {
                document = JsonDocument.Parse(json);
            }
            catch (JsonException ex)
            {
                return (false, null, $"AI inference failed: Local AI server returned invalid JSON: {ex.Message}. Raw response: {TrimForError(json)}");
            }

            using (document)
            {
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
        }

        private async Task<HttpResponseMessage> PostLocalTryOnAsync(
            HttpClient client,
            object request,
            CancellationToken cancellationToken)
        {
            try
            {
                using var content = CreateLocalJsonContent(request);
                return await client.PostAsync($"{_serverManager.ServerUrl}/tryon", content, cancellationToken);
            }
            catch (HttpRequestException)
            {
                await Task.Delay(700, cancellationToken);
                await _serverManager.EnsureServerReadyAsync(cancellationToken);
                using var content = CreateLocalJsonContent(request);
                return await client.PostAsync($"{_serverManager.ServerUrl}/tryon", content, cancellationToken);
            }
            catch (IOException)
            {
                await Task.Delay(700, cancellationToken);
                await _serverManager.EnsureServerReadyAsync(cancellationToken);
                using var content = CreateLocalJsonContent(request);
                return await client.PostAsync($"{_serverManager.ServerUrl}/tryon", content, cancellationToken);
            }
        }

        private static StringContent CreateLocalJsonContent(object request) =>
            new(JsonSerializer.Serialize(request), System.Text.Encoding.UTF8, "application/json");

        private static ByteArrayContent CreateImageContent(byte[] bytes, string fileName)
        {
            var content = new ByteArrayContent(bytes);
            content.Headers.ContentType = new MediaTypeHeaderValue("image/png");
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
            return BuildGarmentDescription(garmentArea, null, null);
        }

        private string BuildGarmentDescription(string garmentArea, string? clothingType)
        {
            return BuildGarmentDescription(garmentArea, clothingType, null);
        }

        private string BuildGarmentDescription(string garmentArea, string? clothingType, string? garmentView)
        {
            var normalizedArea = NormalizeGarmentArea(garmentArea);
            var configuredDescription = _options.HuggingFaceGarmentDescription?.Trim();

            if (!string.IsNullOrWhiteSpace(configuredDescription) &&
                !string.Equals(configuredDescription, "A clothing garment", StringComparison.OrdinalIgnoreCase))
            {
                return configuredDescription;
            }

            var normalizedType = NormalizeClothingType(clothingType);
            var view = NormalizeGarmentView(garmentView);
            var viewText = view == "back" ? "back-facing" : "front-facing";
            var neckText = view == "back" ? "nape to collar, back neck to garment collar" : "neck to neck";
            if (!string.IsNullOrWhiteSpace(normalizedType))
            {
                return normalizedType switch
                {
                    "t-shirt" => $"{viewText} t-shirt worn naturally on the body with realistic fabric drape, preserve the source garment color and printed logo exactly, round collar following the collarbone below the neck, shoulder seam to shoulder seam, sleeves wrapped on arms, fully replace the original upper garment, keep the face and natural neck unchanged, not pasted as a flat sticker",
                    "tank-top" => $"{viewText} sleeveless tank top aligned {neckText}, shoulder strap to shoulder, chest panel to torso",
                    "shirt" => $"{viewText} shirt aligned {neckText}, shoulder to shoulder, elbow to elbow",
                    "chemise" => $"{viewText} chemise blouse aligned {neckText}, shoulder to shoulder, sleeve to arm",
                    "blouse" => $"{viewText} blouse aligned {neckText}, shoulder to shoulder, sleeve to arm, cuff to wrist",
                    "hoodie" => $"{viewText} hoodie aligned {neckText}, shoulder to shoulder, sleeves on arms",
                    "jacket" => $"{viewText} jacket aligned {neckText}, shoulder to shoulder, sleeves on arms",
                    "pants" => "pants aligned waist to waist and knee to knee",
                    "shorts" => "shorts aligned waist to waist and knee to knee",
                    "dress" => $"{viewText} dress aligned {neckText}, shoulder to shoulder, waist to waist, knee to knee",
                    "abaya" => $"{viewText} abaya aligned {neckText}, shoulder to shoulder, elbow to sleeve, full length hem aligned to legs",
                    "galabeya" => $"{viewText} galabeya aligned {neckText}, shoulder to shoulder, elbow to elbow, knee to knee",
                    "jumpsuit" => $"{viewText} child or adult jumpsuit salopette full body outfit aligned {neckText}, shoulder to shoulder, elbow to elbow, waist to waist, knee to knee",
                    _ => $"{normalizedType} clothing garment"
                };
            }

            return normalizedArea switch
            {
                "lower" => "pants or shorts",
                "overall" => "galabeya, dress, or full body outfit",
                _ => "upper body garment"
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
                "abaya" => "abaya",
                "abayas" => "abaya",
                "عباية" => "abaya",
                "عبايات" => "abaya",
                "galabiya" => "galabeya",
                "jellabiya" => "galabeya",
                "jalabiya" => "galabeya",
                "overall" => "jumpsuit",
                "overalls" => "jumpsuit",
                "romper" => "jumpsuit",
                "salopette" => "jumpsuit",
                "salopeit" => "jumpsuit",
                "سالوبيت" => "jumpsuit",
                _ => normalized
            };
        }

        private static string NormalizeGarmentView(string? garmentView)
        {
            var normalized = (garmentView ?? string.Empty)
                .Trim()
                .ToLowerInvariant()
                .Replace(" ", "-")
                .Replace("_", "-");

            return normalized switch
            {
                "back" or "rear" or "behind" or "ضهر" or "ظهر" or "من-ورا" => "back",
                _ => "front"
            };
        }

        private sealed record GradioUploadResult(object FileData, string ApiPrefix);

        private static async Task<GradioUploadResult> UploadGradioFileAsync(
            HttpClient client,
            string baseUrl,
            byte[] imageBytes,
            string fileName,
            string? preferredApiPrefix,
            CancellationToken cancellationToken)
        {
            var prefixes = new[] { preferredApiPrefix, "/gradio_api", string.Empty }
                .Where(prefix => prefix is not null)
                .Select(prefix => prefix!)
                .Distinct(StringComparer.Ordinal)
                .ToArray();

            string? lastJson = null;
            var lastStatusCode = 0;
            foreach (var prefix in prefixes)
            {
                using var content = new MultipartFormDataContent();
                content.Add(CreateGradioUploadContent(imageBytes), "files", fileName);

                using var response = await client.PostAsync($"{baseUrl}{prefix}/upload", content, cancellationToken);
                var json = await response.Content.ReadAsStringAsync(cancellationToken);
                if (!response.IsSuccessStatusCode)
                {
                    lastJson = json;
                    lastStatusCode = (int)response.StatusCode;
                    if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
                    {
                        continue;
                    }

                    throw new InvalidOperationException(BuildHuggingFaceHttpError("upload the images", (int)response.StatusCode, json, baseUrl));
                }

                return new GradioUploadResult(ParseGradioUploadResponse(json, baseUrl, fileName, imageBytes.LongLength), prefix);
            }

            throw new InvalidOperationException(BuildHuggingFaceHttpError("upload the images", lastStatusCode == 0 ? 404 : lastStatusCode, lastJson ?? "{\"detail\":\"Not Found\"}", baseUrl));
        }

        private static object ParseGradioUploadResponse(string json, string baseUrl, string fileName, long fileSize)
        {
            using var document = JsonDocument.Parse(json);
            if (document.RootElement.ValueKind == JsonValueKind.String)
            {
                var uploadedPath = document.RootElement.GetString();
                if (string.IsNullOrWhiteSpace(uploadedPath))
                {
                    throw new InvalidOperationException("Hugging Face upload returned an empty file path.");
                }

                return CreateGradioFileData(uploadedPath, baseUrl, fileName, fileSize);
            }

            var file = document.RootElement;
            if (document.RootElement.ValueKind == JsonValueKind.Object &&
                document.RootElement.TryGetProperty("files", out var files) &&
                files.ValueKind == JsonValueKind.Array)
            {
                var uploadedFiles = files.EnumerateArray().ToList();
                if (uploadedFiles.Count == 0)
                {
                    throw new InvalidOperationException($"Hugging Face upload returned no files. Raw response: {json}");
                }

                file = uploadedFiles[0];
            }

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

                    return CreateGradioFileData(uploadedPath, baseUrl, fileName, fileSize);
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
                TryGetJsonInt64(file, "size") ?? fileSize,
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

        private static string BuildHuggingFaceHttpError(string action, int statusCode, string responseBody, string baseUrl)
        {
            var raw = TrimForError(responseBody);
            var spaceUrl = BuildHuggingFaceSpacePageUrl(baseUrl);

            if (statusCode == 503 || responseBody.Contains("space is in error", StringComparison.OrdinalIgnoreCase))
            {
                if (baseUrl.Contains("yisol-idm-vton", StringComparison.OrdinalIgnoreCase))
                {
                    return $"The configured Hugging Face Space is the old public demo ({spaceUrl}), and Hugging Face says it is currently in error. Replace VirtualTryOn:HuggingFaceSpaceUrl with your own duplicated Space URL, then try again. Raw response: {raw}";
                }

                return $"The public Hugging Face try-on Space is currently unavailable while trying to {action}. Hugging Face reports that the Space is in error. Check {spaceUrl} or switch VirtualTryOn:Mode to another backend. Raw response: {raw}";
            }

            return $"Hugging Face could not {action}. HTTP {statusCode}. Raw response: {raw}";
        }

        private static string BuildHuggingFaceSpacePageUrl(string baseUrl)
        {
            if (!Uri.TryCreate(baseUrl, UriKind.Absolute, out var uri))
            {
                return "the Hugging Face Space page";
            }

            var host = uri.Host;
            const string suffix = ".hf.space";
            if (!host.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
            {
                return "the Hugging Face Space page";
            }

            var spaceHost = host[..^suffix.Length];
            var separatorIndex = spaceHost.IndexOf('-');
            if (separatorIndex <= 0 || separatorIndex >= spaceHost.Length - 1)
            {
                return "the Hugging Face Space page";
            }

            var owner = spaceHost[..separatorIndex];
            var spaceName = spaceHost[(separatorIndex + 1)..];
            return $"https://huggingface.co/spaces/{owner}/{spaceName}";
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

        private static byte[] ResizeImageForInference(byte[] imageBytes, int maxSide, long quality)
        {
            try
            {
                using var input = new MemoryStream(imageBytes);
                using var source = Image.FromStream(input);
                var largestSide = Math.Max(source.Width, source.Height);
                if (largestSide <= maxSide && imageBytes.Length <= 1_500_000)
                {
                    return imageBytes;
                }

                var scale = Math.Min(1.0, maxSide / (double)largestSide);
                var targetWidth = Math.Max(1, (int)Math.Round(source.Width * scale));
                var targetHeight = Math.Max(1, (int)Math.Round(source.Height * scale));

                using var bitmap = new Bitmap(targetWidth, targetHeight);
                using (var graphics = Graphics.FromImage(bitmap))
                {
                    graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                    graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                    graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                    graphics.DrawImage(source, 0, 0, targetWidth, targetHeight);
                }

                using var output = new MemoryStream();
                var encoder = ImageCodecInfo.GetImageEncoders()
                    .FirstOrDefault(codec => string.Equals(codec.MimeType, "image/jpeg", StringComparison.OrdinalIgnoreCase));

                if (encoder == null)
                {
                    bitmap.Save(output, ImageFormat.Jpeg);
                }
                else
                {
                    using var encoderParameters = new EncoderParameters(1);
                    encoderParameters.Param[0] = new EncoderParameter(Encoder.Quality, quality);
                    bitmap.Save(output, encoder, encoderParameters);
                }

                return output.ToArray();
            }
            catch
            {
                return imageBytes;
            }
        }

        private static string NormalizeGarmentArea(string garmentArea)
        {
            return garmentArea.Trim().ToLowerInvariant() switch
            {
                "upper_body" => "upper",
                "top" => "upper",
                "tank-top" => "upper",
                "tanktop" => "upper",
                "vest" => "upper",
                "sleeveless" => "upper",
                "chemise" => "upper",
                "chemise-shirt" => "upper",
                "blouse" => "upper",
                "blouses" => "upper",
                "lower_body" => "lower",
                "pants" => "lower",
                "trousers" => "lower",
                "short" => "lower",
                "shorts" => "lower",
                "dress" => "overall",
                "dresses" => "overall",
                "jumpsuit" => "overall",
                "overall" => "overall",
                "overalls" => "overall",
                "romper" => "overall",
                "salopette" => "overall",
                "salopeit" => "overall",
                "سالوبيت" => "overall",
                "abaya" => "overall",
                "abayas" => "overall",
                "عباية" => "overall",
                "عبايات" => "overall",
                "galabeya" => "overall",
                "galabiya" => "overall",
                "jellabiya" => "overall",
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

        private static string BuildApiErrorMessage(string apiUrl, int statusCode, string responseText)
        {
            if (statusCode == 530 &&
                responseText.Contains("1033", StringComparison.OrdinalIgnoreCase))
            {
                return $"Colab/Cloudflare tunnel is not reachable. The saved API URL is down or expired: {apiUrl}. Keep the Colab notebook cell running, copy the new URL ending with /tryon, then paste it in the Upload page or update VirtualTryOn:ApiUrl.";
            }

            return $"API returned HTTP {statusCode}: {responseText}";
        }

        private static string BuildInferenceExceptionMessage(Exception exception)
        {
            var message = exception.Message;
            if (exception.InnerException != null &&
                !string.Equals(exception.InnerException.Message, message, StringComparison.Ordinal))
            {
                message += $" Inner error: {exception.InnerException.Message}";
            }

            if (message.Contains("copying content to a stream", StringComparison.OrdinalIgnoreCase))
            {
                message += " This usually means the local AI server closed the connection while ASP.NET was sending the images. Check that the Python inference server is still running and that the selected images are valid.";
            }

            if (message.Contains("No such host is known", StringComparison.OrdinalIgnoreCase) ||
                message.Contains("trycloudflare.com", StringComparison.OrdinalIgnoreCase))
            {
                message += " The Colab/Cloudflare tunnel URL is expired or not running. Start the Colab API cell again, copy the new /tryon URL, and paste it in the Upload page.";
            }

            return $"AI inference failed: {message}";
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
