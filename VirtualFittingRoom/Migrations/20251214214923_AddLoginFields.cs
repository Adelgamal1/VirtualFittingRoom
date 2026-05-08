using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace VirtualFittingRoom.Migrations
{
    public partial class AddLoginFields : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "PantsLength",
                table: "UserMeasurements");

            migrationBuilder.DropColumn(
                name: "TopLength",
                table: "UserMeasurements");

            migrationBuilder.AddColumn<string>(
                name: "ConfirmPassword",
                table: "UserMeasurements",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "Email",
                table: "UserMeasurements",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "Password",
                table: "UserMeasurements",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ConfirmPassword",
                table: "UserMeasurements");

            migrationBuilder.DropColumn(
                name: "Email",
                table: "UserMeasurements");

            migrationBuilder.DropColumn(
                name: "Password",
                table: "UserMeasurements");

            migrationBuilder.AddColumn<float>(
                name: "PantsLength",
                table: "UserMeasurements",
                type: "real",
                nullable: false,
                defaultValue: 0f);

            migrationBuilder.AddColumn<float>(
                name: "TopLength",
                table: "UserMeasurements",
                type: "real",
                nullable: false,
                defaultValue: 0f);
        }
    }
}
