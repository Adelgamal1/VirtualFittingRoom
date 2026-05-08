using System;
using System.ComponentModel.DataAnnotations;

namespace VirtualFittingRoom.Models
{
    public class UserImage
    {
        public int Id { get; set; }

        // 🔗 Foreign Key
        public int UserMeasurementId { get; set; }

        // 🔗 Navigation Property
        public UserMeasurement UserMeasurement { get; set; }

        // 🖼 Image Data
        [Required]
        public byte[] ImageData { get; set; }

        [Required]
        public string ImageType { get; set; }

        // ⭐ Rating
        public string? Rating { get; set; }

        // 🕒 Created Date
        public DateTime CreatedAt { get; set; } = DateTime.Now;
    }
}
