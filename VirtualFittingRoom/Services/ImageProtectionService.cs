using Microsoft.AspNetCore.DataProtection;

namespace VirtualFittingRoom.Services
{
    public class ImageProtectionService
    {
        private readonly IDataProtector _protector;

        public ImageProtectionService(IDataProtectionProvider provider)
        {
            _protector = provider.CreateProtector("VirtualFittingRoom.UserImage.v1");
        }

        public byte[] Protect(byte[] imageBytes)
        {
            return _protector.Protect(imageBytes);
        }

        public byte[] Unprotect(byte[] storedBytes)
        {
            try
            {
                return _protector.Unprotect(storedBytes);
            }
            catch
            {
                return storedBytes;
            }
        }
    }
}
