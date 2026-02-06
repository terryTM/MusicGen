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
