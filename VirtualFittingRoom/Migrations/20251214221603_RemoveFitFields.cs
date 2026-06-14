using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace VirtualFittingRoom.Migrations
{
    public partial class RemoveFitFields : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "PantsFit",
                table: "UserMeasurements");

            migrationBuilder.DropColumn(
                name: "TopFit",
                table: "UserMeasurements");
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "PantsFit",
                table: "UserMeasurements",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "TopFit",
                table: "UserMeasurements",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");
        }
    }
}
