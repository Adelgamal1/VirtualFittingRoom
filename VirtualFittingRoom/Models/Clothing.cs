namespace VirtualFittingRoom.Models
{
    public class Clothing
    {
        public int Id { get; set; }
        public string Gender { get; set; }      // Male / Female
        public string Name { get; set; }        // Shirt, Pants, Jacket...
        public string ImagePath { get; set; }   // /images/xxx.png
    }
}
