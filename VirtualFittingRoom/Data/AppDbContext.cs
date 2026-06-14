using Microsoft.EntityFrameworkCore;
using VirtualFittingRoom.Models;

namespace VirtualFittingRoom.Data
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options)
            : base(options)
        {
        }

        public DbSet<UserMeasurement> UserMeasurements { get; set; }
        public DbSet<UserImage> UserImages { get; set; }
        public DbSet<Clothing> Clothes { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)

        {
            base.OnModelCreating(modelBuilder);

            modelBuilder.Entity<UserMeasurement>()
                .HasIndex(u => u.Email)
                .IsUnique();

            modelBuilder.Entity<UserImage>()
                .HasOne(i => i.UserMeasurement)   // 👈 Navigation Property
                .WithMany()
                .HasForeignKey(i => i.UserMeasurementId)
                .OnDelete(DeleteBehavior.Cascade);
        }

    }
}
