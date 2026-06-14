# Social login setup

The Google, Facebook, and Apple buttons are already wired to the app. They will only start the real provider login after valid OAuth credentials are added.

## Local URL

This project currently runs on:

```text
http://localhost:5001
```

## Callback URLs

Add these exact redirect/callback URLs in each provider dashboard:

```text
Google:   http://localhost:5001/signin-google
Facebook: http://localhost:5001/signin-facebook
Apple:    https://your-real-domain.com/signin-apple
```

Apple does not allow `localhost` or plain `http` for web redirect URIs, so Apple sign-in needs a real HTTPS domain or an HTTPS tunnel during development.

## Store secrets locally

From the `VirtualFittingRoom` project folder, run:

```powershell
dotnet user-secrets set "Authentication:Google:ClientId" "YOUR_GOOGLE_CLIENT_ID"
dotnet user-secrets set "Authentication:Google:ClientSecret" "YOUR_GOOGLE_CLIENT_SECRET"

dotnet user-secrets set "Authentication:Facebook:AppId" "YOUR_FACEBOOK_APP_ID"
dotnet user-secrets set "Authentication:Facebook:AppSecret" "YOUR_FACEBOOK_APP_SECRET"

dotnet user-secrets set "Authentication:Apple:ClientId" "YOUR_APPLE_SERVICES_ID"
dotnet user-secrets set "Authentication:Apple:ClientSecret" "YOUR_APPLE_CLIENT_SECRET_JWT"
```

Restart the app after setting or changing secrets.

## Provider notes

Google:
- Create an OAuth client of type Web application.
- Add `http://localhost:5001/signin-google` to Authorized redirect URIs.

Facebook:
- Create a Meta app and add Facebook Login.
- Add `http://localhost:5001/signin-facebook` to Valid OAuth Redirect URIs.
- Make sure the app can request `email` and `public_profile`.

Apple:
- Configure Sign in with Apple for a Services ID.
- Add your production HTTPS domain and return URL.
- The client secret is a JWT generated from your Apple private key, not a normal password.
