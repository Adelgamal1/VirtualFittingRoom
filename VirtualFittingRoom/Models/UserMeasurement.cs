using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore;

namespace VirtualFittingRoom.Models
{
    [Index(nameof(Email), IsUnique = true)]
    public class UserMeasurement
    {
        public int Id { get; set; }

        // ===== Account Info =====

        [Required(ErrorMessage = "Full name is required")]
        public string FullName { get; set; }

        [Required(ErrorMessage = "Email is required")]
        [EmailAddress(ErrorMessage = "Invalid email format")]
        public string Email { get; set; }

        // 🆕 Phone Number
        [Required(ErrorMessage = "Phone number is required")]
        [Phone(ErrorMessage = "Invalid phone number")]
        public string PhoneNumber { get; set; }

        // ✅ Stored password (hashed)
        public string PasswordHash { get; set; }
        // ===== Password Reset =====

        public string? ResetToken { get; set; }
        public DateTime? ResetTokenExpiry { get; set; }
        // ===== Not Stored =====
        [NotMapped]
        [Required(ErrorMessage = "Password is required")]
        [MinLength(6, ErrorMessage = "Password must be at least 6 characters")]
        public string Password { get; set; }

        [NotMapped]
        [Required(ErrorMessage = "Confirm password is required")]
        [Compare("Password", ErrorMessage = "Passwords do not match")]
        public string ConfirmPassword { get; set; }

        // ===== Body Measurements =====
        [Required(ErrorMessage = "Age is required")]
        public int Age { get; set; }

        [Required(ErrorMessage = "Weight is required")]
        public float Weight { get; set; }

        [Required(ErrorMessage = "Height is required")]
        public float Height { get; set; }

        [Required(ErrorMessage = "Gender is required")]
        public string Gender { get; set; }


    }
}
