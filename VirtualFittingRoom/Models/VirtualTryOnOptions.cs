namespace VirtualFittingRoom.Models
{
    public class VirtualTryOnOptions
    {
        public string Mode { get; set; } = "Api";
        public string PythonExecutable { get; set; } = "python";
        public string InferenceScriptPath { get; set; } = string.Empty;
        public string ServerScriptPath { get; set; } = string.Empty;
        public string WorkingDirectory { get; set; } = string.Empty;
        public string ServerUrl { get; set; } = "http://127.0.0.1:5011";
        public int ServerStartupTimeoutSeconds { get; set; } = 120;
        public string ApiUrl { get; set; } = string.Empty;
        public string ApiKey { get; set; } = string.Empty;
        public string ApiKeyHeader { get; set; } = "Authorization";
        public string ApiPersonFieldName { get; set; } = "person_image";
        public string ApiClothingFieldName { get; set; } = "garment_image";
        public string ApiCategoryFieldName { get; set; } = "category";
        public string ApiResponseImageField { get; set; } = "outputImageBase64";
        public int ApiTimeoutSeconds { get; set; } = 600;
        public string ReplicateApiToken { get; set; } = string.Empty;
        public string ReplicateModelOwner { get; set; } = "cedoysch";
        public string ReplicateModelName { get; set; } = "flux-fill-redux-try-on";
        public string ReplicateVersion { get; set; } = "cf5cb07a25e726fe2fac166a8c5ab52ddccd48657741670fb09d9954d4d8446f";
        public string ReplicatePersonFieldName { get; set; } = "person_image";
        public string ReplicateClothingFieldName { get; set; } = "cloth_image";
        public string ReplicateCategoryFieldName { get; set; } = "cloth_type";
        public string ReplicateOutputFormat { get; set; } = "png";
        public int ReplicateWaitSeconds { get; set; } = 60;
        public int ReplicatePollSeconds { get; set; } = 3;
        public string HuggingFaceSpaceUrl { get; set; } = "https://yisol-idm-vton.hf.space";
        public string HuggingFaceApiName { get; set; } = "tryon";
        public string HuggingFaceToken { get; set; } = string.Empty;
        public int HuggingFaceDenoiseSteps { get; set; } = 30;
        public int HuggingFaceSeed { get; set; } = 42;
        public bool HuggingFaceAutoMask { get; set; } = true;
        public bool HuggingFaceAutoCrop { get; set; } = false;
        public string HuggingFaceGarmentDescription { get; set; } = "A clothing garment";
        public string ArgumentsTemplate { get; set; } =
            "\"{script}\" --person \"{person}\" --cloth \"{cloth}\" --category {category} --output \"{output}\"";
    }
}
