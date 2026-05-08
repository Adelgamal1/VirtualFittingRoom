using System.Diagnostics;
using Microsoft.Extensions.Options;
using VirtualFittingRoom.Models;

namespace VirtualFittingRoom.Services
{
    public class LocalInferenceServerManager
    {
        private readonly VirtualTryOnOptions _options;
        private readonly HttpClient _httpClient;
        private readonly SemaphoreSlim _startupLock = new(1, 1);
        private Process? _serverProcess;

        public LocalInferenceServerManager(IOptions<VirtualTryOnOptions> options, IHttpClientFactory httpClientFactory)
        {
            _options = options.Value;
            _httpClient = httpClientFactory.CreateClient();
        }

        public string ServerUrl => _options.ServerUrl.TrimEnd('/');

        public async Task<(bool Success, string? Error)> EnsureServerReadyAsync(CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(_options.ServerScriptPath) || !File.Exists(_options.ServerScriptPath))
            {
                return (false, "AI server script path is not configured yet. Update VirtualTryOn:ServerScriptPath in appsettings.json.");
            }

            if (await IsHealthyAsync(cancellationToken))
            {
                return (true, null);
            }

            await _startupLock.WaitAsync(cancellationToken);
            try
            {
                if (await IsHealthyAsync(cancellationToken))
                {
                    return (true, null);
                }

                if (_serverProcess is null || _serverProcess.HasExited)
                {
                    StartServerProcess();
                }

                var timeoutAt = DateTime.UtcNow.AddSeconds(Math.Max(10, _options.ServerStartupTimeoutSeconds));
                while (DateTime.UtcNow < timeoutAt)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    if (await IsHealthyAsync(cancellationToken))
                    {
                        return (true, null);
                    }

                    if (_serverProcess is { HasExited: true })
                    {
                        return (false, "AI server process exited before it became ready. Check the Python model environment and script paths.");
                    }

                    await Task.Delay(1000, cancellationToken);
                }

                return (false, "AI server took too long to start. The model may still be loading or the machine may be low on memory.");
            }
            finally
            {
                _startupLock.Release();
            }
        }

        private void StartServerProcess()
        {
            var workingDirectory = !string.IsNullOrWhiteSpace(_options.WorkingDirectory)
                ? _options.WorkingDirectory
                : Path.GetDirectoryName(_options.ServerScriptPath)!;

            Directory.CreateDirectory(workingDirectory);

            var serverUri = new Uri(_options.ServerUrl);
            var host = serverUri.Host;
            var port = serverUri.Port;

            var arguments = string.Join(" ",
                Quote(_options.ServerScriptPath),
                "--host", Quote(host),
                "--port", port.ToString(),
                "--project-root", Quote(_options.WorkingDirectory),
                "--base-model-path", Quote(ExtractArgumentValue("--base-model-path")),
                "--resume-path", Quote(ExtractArgumentValue("--resume-path")),
                "--attn-version", Quote(ExtractArgumentValue("--attn-version", "mix")));

            var startInfo = new ProcessStartInfo
            {
                FileName = _options.PythonExecutable,
                Arguments = arguments,
                WorkingDirectory = workingDirectory,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = false,
                RedirectStandardError = false
            };

            _serverProcess = new Process
            {
                StartInfo = startInfo,
                EnableRaisingEvents = true
            };
            _serverProcess.Start();
        }

        private async Task<bool> IsHealthyAsync(CancellationToken cancellationToken)
        {
            try
            {
                using var response = await _httpClient.GetAsync($"{ServerUrl}/health", cancellationToken);
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }

        private string ExtractArgumentValue(string key, string fallback = "")
        {
            var template = _options.ArgumentsTemplate ?? string.Empty;
            var token = $"{key} ";
            var index = template.IndexOf(token, StringComparison.OrdinalIgnoreCase);
            if (index < 0)
            {
                return fallback;
            }

            index += token.Length;
            if (index >= template.Length)
            {
                return fallback;
            }

            if (template[index] == '"')
            {
                index++;
                var endQuote = template.IndexOf('"', index);
                return endQuote > index
                    ? template.Substring(index, endQuote - index)
                    : fallback;
            }

            var end = template.IndexOf(' ', index);
            return end > index
                ? template.Substring(index, end - index)
                : template.Substring(index);
        }

        private static string Quote(string value) => $"\"{value}\"";
    }
}
