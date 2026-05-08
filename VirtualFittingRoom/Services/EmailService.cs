using System.Net;
using System.Net.Mail;

namespace VirtualFittingRoom.Services
{
    public static class EmailService
    {
        public static void Send(string toEmail, string subject, string body)
        {
            var fromEmail = Environment.GetEnvironmentVariable("VFR_EMAIL_FROM");
            var appPassword = Environment.GetEnvironmentVariable("VFR_EMAIL_APP_PASSWORD");

            if (string.IsNullOrWhiteSpace(fromEmail) || string.IsNullOrWhiteSpace(appPassword))
            {
                throw new InvalidOperationException(
                    "Email settings are missing. Set VFR_EMAIL_FROM and VFR_EMAIL_APP_PASSWORD environment variables.");
            }

            var message = new MailMessage();
            message.From = new MailAddress(fromEmail, "Virtual Fitting Room");
            message.To.Add(toEmail);
            message.Subject = subject;
            message.Body = body;
            message.IsBodyHtml = true;

            var smtp = new SmtpClient("smtp.gmail.com", 587);
            smtp.Credentials = new NetworkCredential(fromEmail, appPassword);
            smtp.EnableSsl = true;

            smtp.Send(message);
        }
    }
}

