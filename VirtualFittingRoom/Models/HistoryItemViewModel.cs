namespace VirtualFittingRoom.Models
{
    public class HistoryItemViewModel
    {
        public int Id { get; set; }
        public string ImageUrl { get; set; } = string.Empty;
        public string ImageType { get; set; } = string.Empty;
        public string? Rating { get; set; }
        public DateTime CreatedAt { get; set; }
    }
}
