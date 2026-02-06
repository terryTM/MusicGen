import express from "express";
import cors from "cors";
import { fileURLToPath } from "url";
import { dirname } from "path";
import dotenv from "dotenv";
import ytpl from "ytpl";

dotenv.config();

const app = express();
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

// Extract playlist ID from YouTube Music link
function extractPlaylistId(url) {
  // Handles: https://music.youtube.com/playlist?list=PLxxxxxx
  const match = url.match(/list=([a-zA-Z0-9_-]+)/);
  return match ? match[1] : null;
}

// Fetch YouTube Music playlist
async function getYouTubePlaylist(playlistId) {
  try {
    // Fetch playlist using ytpl
    const playlist = await ytpl(playlistId, { limit: Infinity });

    const tracks = playlist.items.map((item) => {
      return {
        name: item.title || "Unknown",
        artists: item.author?.name || "Unknown Artist",
        duration: item.duration || "0:00",
        url: item.url,
        thumbnail: item.thumbnail,
      };
    });

    return {
      name: playlist.title,
      description: playlist.description || "",
      owner: playlist.estimatedItemCount ? `${playlist.estimatedItemCount} items` : "Unknown",
      image: playlist.bestThumbnail?.url || null,
      tracks,
      totalTracks: tracks.length,
    };
  } catch (error) {
    throw new Error(`Failed to fetch YouTube playlist: ${error.message}`);
  }
}

// Get YouTube Music playlist data
app.get("/api/playlist/:id", async (req, res) => {
  try {
    const { id } = req.params;
    
    if (!id) {
      return res.status(400).json({ error: "No playlist ID provided" });
    }

    const data = await getYouTubePlaylist(id);
    res.json(data);
  } catch (error) {
    console.error("Playlist fetch error:", error);
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🎵 YouTube Music Server running on http://localhost:${PORT}`);
});

// --- Deezer preview support ---
// Search Deezer for a track by title + artist and return preview URL
async function searchDeezerPreview(title, artist) {
  function clean(s) {
    if (!s) return s;
    // remove parenthetical/framed phrases and common noise
    return s.replace(/\([^)]*\)/g, "")
            .replace(/\[[^\]]*\]/g, "")
            .replace(/"/g, "")
            .replace(/official video|official music video|official|music video|video|lyrics|audio|HD|\bft\.?\b/ig, "")
            .replace(/\s+/g, " ")
            .trim();
  }

  const cleanedTitle = clean(title);
  const cleanedArtist = clean(artist);

  const queries = [
    `track:\"${title}\" artist:\"${artist}\"`,
    `track:\"${cleanedTitle}\" artist:\"${cleanedArtist}\"`,
    `track:\"${cleanedTitle}\"`,
    `track:\"${title}\"`,
  ];

  for (const q of queries) {
    try {
      const url = `https://api.deezer.com/search?q=${encodeURIComponent(q)}`;
      const resp = await fetch(url);
      if (!resp.ok) continue;
      const data = await resp.json();
      const first = data.data && data.data.length ? data.data[0] : null;
      if (!first) continue;

      return {
        deezer_id: first.id,
        title: first.title,
        artist: first.artist?.name || artist,
        preview: first.preview || null,
        link: first.link || null,
      };
    } catch (e) {
      // try next query
      console.debug('Deezer query failed for', q, e.message);
      continue;
    }
  }

  return null;
}

// Given a YouTube playlist id, return tracks with Deezer preview URLs
app.get('/api/playlist/:id/deezer-previews', async (req, res) => {
  try {
    const { id } = req.params;
    if (!id) return res.status(400).json({ error: 'No playlist ID provided' });

    const yt = await getYouTubePlaylist(id);
    const out = [];

    // sequentially search Deezer for each track to avoid overwhelming API
    for (const t of yt.tracks) {
      const title = t.name;
      const artist = Array.isArray(t.artists) ? t.artists.join(' ') : t.artists;
      try {
        const found = await searchDeezerPreview(title, artist);
        out.push({
          name: title,
          artists: artist,
          duration: t.duration,
          deezer: found,
        });
      } catch (e) {
        out.push({ name: title, artists: artist, duration: t.duration, deezer: null, error: e.message });
      }
    }

    res.json({ playlist: { name: yt.name, totalTracks: yt.totalTracks }, tracks: out });
  } catch (error) {
    console.error('Deezer preview error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Proxy a Deezer preview URL through the server (avoids CORS and direct leaking)
app.get('/api/preview-proxy', async (req, res) => {
  const { url } = req.query;
  if (!url) return res.status(400).send('Missing url param');

  try {
    const decoded = decodeURIComponent(url);
    const resp = await fetch(decoded);
    if (!resp.ok) return res.status(502).send('Failed to fetch preview');

    const arrayBuffer = await resp.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    res.setHeader('content-type', resp.headers.get('content-type') || 'audio/mpeg');
    res.setHeader('content-length', buffer.length);
    res.end(buffer);
  } catch (e) {
    console.error('Preview proxy error:', e);
    res.status(500).send('Proxy error');
  }
});
