# MusicGen - Spotify Playlist Access

A simple web app that lets users log in with Spotify and view their playlists.

## Setup

### 1. Get Spotify Credentials

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Log in or create an account
3. Create a new app
4. Accept the terms and create
5. You'll get:
   - **Client ID**
   - **Client Secret** (keep this secret!)
6. Go to "Edit Settings" and add your Redirect URI:
   - For local development: `http://localhost:3000/callback`
   - For production: `https://yourdomain.com/callback`

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
REDIRECT_URI=http://localhost:3000/callback
PORT=3000
```

### 3. Install Dependencies

```bash
npm install
```

### 4. Run Server

```bash
npm start
```

Server will start at `http://localhost:3000`

### 5. Test

Open `http://localhost:3000` in your browser and click the green Spotify button!

## How It Works

1. **Frontend** - User clicks the green "Login with Spotify" button
2. **Backend** (`/api/login`) - Redirects to Spotify authorization page
3. **Spotify** - User authorizes the app
4. **Backend** (`/callback`) - Spotify redirects back with auth code
5. **Backend** - Exchanges code for access token
6. **Backend** - Fetches user profile & playlists
7. **Frontend** - Shows success page with user info and playlist list

## Deployment

### Heroku

```bash
heroku login
heroku create your-app-name
heroku config:set SPOTIFY_CLIENT_ID=your_id
heroku config:set SPOTIFY_CLIENT_SECRET=your_secret
heroku config:set REDIRECT_URI=https://your-app-name.herokuapp.com/callback
git push heroku main
```

### Vercel

Vercel doesn't support persistent backends, so for a serverless solution you'd need to restructure this code. For now, use a traditional server like Heroku or DigitalOcean.

### Manual Server (DigitalOcean, AWS, etc.)

1. Push code to GitHub
2. SSH into your server
3. Clone the repo
4. Set environment variables
5. Run `npm start`

## Security Notes

⚠️ **Never commit `.env` to GitHub** - it's in `.gitignore`

Never expose your Client Secret in browser code. This backend keeps it secure on the server.

## File Structure

```
.
├── index.html          # Frontend (simple login button)
├── server.js           # Express backend (OAuth handler)
├── package.json        # Dependencies
├── .env                # Environment variables (not committed)
└── .env.example        # Example env file
```

## Customization

To add more features:
- Edit the `scopes` in `server.js` to request more permissions
- Modify `/callback` to save user data to a database
- Build an API to fetch track info, create playlists, etc.

---

Questions? Check the [Spotify Web API docs](https://developer.spotify.com/documentation/web-api)
