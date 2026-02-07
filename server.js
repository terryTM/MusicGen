import express from "express";
import cors from "cors";
import { fileURLToPath } from "url";
import { dirname } from "path";
import dotenv from "dotenv";
import ytpl from "ytpl";
import https from "https";
import http from "http";

dotenv.config();

const app = express();
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname, {
  setHeaders: (res, path) => {
    if (path.endsWith('.js')) {
      res.setHeader('Content-Type', 'application/javascript');
    }
  }
}));
app.use('/node_modules', express.static(__dirname + '/node_modules', {
  setHeaders: (res, path) => {
    if (path.endsWith('.js')) {
      res.setHeader('Content-Type', 'application/javascript');
    }
  }
}));

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

// Trippify visualizer page route
app.get('/trippify', (req, res) => {
  res.sendFile(__dirname + '/trippify.html');
});

// Trippify song page route
app.get('/trippify_song', (req, res) => {
  res.sendFile(__dirname + '/trippify_song.html');
});

// Audio proxy endpoint to bypass CORS
app.get('/api/proxy-audio', async (req, res) => {
  const audioUrl = req.query.url;
  
  if (!audioUrl) {
    return res.status(400).json({ error: 'No URL provided' });
  }

  try {
    const urlObj = new URL(audioUrl);
    const protocol = urlObj.protocol === 'https:' ? https : http;

    protocol.get(audioUrl, (proxyRes) => {
      // Forward headers
      res.setHeader('Content-Type', proxyRes.headers['content-type'] || 'audio/mpeg');
      res.setHeader('Access-Control-Allow-Origin', '*');
      
      if (proxyRes.headers['content-length']) {
        res.setHeader('Content-Length', proxyRes.headers['content-length']);
      }
      
      // Enable range requests for seeking
      if (proxyRes.headers['accept-ranges']) {
        res.setHeader('Accept-Ranges', proxyRes.headers['accept-ranges']);
      }

      // Pipe the audio stream
      proxyRes.pipe(res);
    }).on('error', (err) => {
      console.error('Proxy error:', err);
      res.status(500).json({ error: 'Failed to fetch audio' });
    });
  } catch (error) {
    console.error('Invalid URL:', error);
    res.status(400).json({ error: 'Invalid URL' });
  }
});

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

// Parse a YouTube video title into { trackName, artists[] }
function parseYouTubeTitle(rawTitle, channelName) {
  let title = rawTitle || "";

  // Remove parenthetical/bracketed noise: (Official Video), [HD], etc.
  title = title.replace(/\([^)]*\)/g, "").replace(/\[[^\]]*\]/g, "");

  // Remove pipe-separated suffixes like "| EL ÚLTIMO TOUR DEL MUNDO"
  title = title.replace(/\|.*$/, "");

  // Remove common YouTube noise words
  title = title.replace(/official\s*video|official\s*music\s*video|official\s*audio|music\s*video|lyric\s*video|official|lyrics|audio|video|HD|HQ|4K|visualizer/gi, "");

  // Remove credit tags like "🎥By. Ryan Lynch", "Prod. By ..."
  title = title.replace(/🎥.*$/g, "");
  title = title.replace(/\bprod\.?\s*by\.?\s.*/gi, "");
  title = title.replace(/\bdir\.?\s*by\.?\s.*/gi, "");

  // Remove stray quotes
  title = title.replace(/"/g, "");

  title = title.replace(/\s+/g, " ").trim();

  // Split on " - " or " – " to separate artist(s) from track name
  let trackName = title;
  let parsedArtists = [];

  const dashMatch = title.match(/^(.+?)\s*[-–]\s+(.+)$/);
  if (dashMatch) {
    const left = dashMatch[1].trim();   // artist side
    const right = dashMatch[2].trim();  // track side

    // Split artists on " x ", " X ", " & ", ", ", "feat.", "ft.", "featuring"
    parsedArtists = left.split(/\s+x\s+|\s*&\s*|\s*,\s*|\s+feat\.?\s+|\s+ft\.?\s+|\s+featuring\s+/i).map(a => a.trim()).filter(Boolean);
    trackName = right;
  }

  // Strip featured artist tags from the track name and capture featured artists
  const featMatch = trackName.match(/\s*(ft\.?|feat\.?|featuring)\s+(.+)/i);
  if (featMatch) {
    const featArtists = featMatch[2].split(/\s*&\s*|\s*,\s*|\s+and\s+/i).map(a => a.trim()).filter(Boolean);
    parsedArtists.push(...featArtists);
    trackName = trackName.replace(/\s*(ft\.?|feat\.?|featuring)\s+.*/i, "").trim();
  }

  // Clean up channel name for fallback
  const cleanChannel = channelName
    ? channelName.replace(/\s*-\s*Topic$/, "").replace(/\s*Official$/i, "").trim()
    : "";

  // If no dash split happened, try stripping the channel/artist name from the start of the title
  // Handles: 'Lil Baby "Freestyle" Official Music Video' → trackName = "Freestyle"
  if (parsedArtists.length === 0 && cleanChannel) {
    const channelLower = cleanChannel.toLowerCase();
    const trackLower = trackName.toLowerCase();
    if (trackLower.startsWith(channelLower + " ")) {
      trackName = trackName.substring(cleanChannel.length).trim();
      parsedArtists = [cleanChannel];
    } else {
      parsedArtists = [cleanChannel];
    }
  }

  return { trackName, artists: parsedArtists };
}

// Normalize a string for fuzzy comparison
function normalize(s) {
  return (s || "").toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "")  // strip accents
    .replace(/[^a-z0-9\s]/g, "")                        // strip punctuation
    .replace(/\s+/g, " ")
    .trim();
}

// Check if two strings are a fuzzy match (one contains the other, or high overlap)
function fuzzyMatch(a, b) {
  const na = normalize(a);
  const nb = normalize(b);
  if (!na || !nb) return false;
  // Direct containment
  if (na.includes(nb) || nb.includes(na)) return true;
  // Check if all words from the shorter string appear in the longer one
  const shorter = na.length <= nb.length ? na : nb;
  const longer = na.length <= nb.length ? nb : na;
  const words = shorter.split(" ").filter(w => w.length > 1);
  const matched = words.filter(w => longer.includes(w));
  return words.length > 0 && matched.length / words.length >= 0.8;
}

// Reject results that are covers, karaoke, instrumentals, etc.
const JUNK_PATTERNS = /\b(karaoke|instrumental|originally performed|8[- ]?bit|tribute|cover|lullab|music box|acoustic version by|piano version|made famous|in the style of|parody|emulation|babies|study|meditation)\b/i;

function isJunkResult(result) {
  const title = result.title || "";
  const artist = result.artist?.name || "";
  const album = result.album?.title || "";
  return JUNK_PATTERNS.test(title) || JUNK_PATTERNS.test(artist) || JUNK_PATTERNS.test(album);
}

// Score a Deezer result against expected track/artist
function scoreResult(result, expectedTrack, expectedArtists) {
  let score = 0;
  const rTitle = result.title || "";
  const rArtist = result.artist?.name || "";

  // Title match
  if (fuzzyMatch(rTitle, expectedTrack)) score += 50;

  // Artist match - check against any of the expected artists
  for (const ea of expectedArtists) {
    if (fuzzyMatch(rArtist, ea)) { score += 40; break; }
  }

  // Bonus: has a preview URL
  if (result.preview) score += 10;

  return score;
}

// Search Deezer for a track - scan results and pick the best non-junk match
async function searchDeezerPreview(title, artist) {
  const { trackName, artists } = parseYouTubeTitle(title, artist);
  const primaryArtist = artists[0] || artist || "";

  // Build a prioritized list of search queries
  const queries = [
    `track:"${trackName}" artist:"${primaryArtist}"`,
    `${trackName} ${primaryArtist}`,
    `track:"${trackName}"`,
  ];

  // De-duplicate queries
  const seen = new Set();
  const uniqueQueries = queries.filter(q => { const k = q.toLowerCase(); if (seen.has(k)) return false; seen.add(k); return true; });

  for (const q of uniqueQueries) {
    try {
      const url = `https://api.deezer.com/search?q=${encodeURIComponent(q)}&limit=25`;
      const resp = await fetch(url);
      if (!resp.ok) continue;
      const data = await resp.json();
      if (!data.data || data.data.length === 0) continue;

      // Score and rank all non-junk results
      let bestResult = null;
      let bestScore = -1;

      for (const r of data.data) {
        if (isJunkResult(r)) continue;
        const s = scoreResult(r, trackName, artists);
        if (s > bestScore) {
          bestScore = s;
          bestResult = r;
        }
      }

      // Require a minimum score (at least title OR artist should match)
      if (bestResult && bestScore >= 40) {
        return {
          deezer_id: bestResult.id,
          title: bestResult.title,
          artist: bestResult.artist?.name || artist,
          preview: bestResult.preview || null,
          link: bestResult.link || null,
        };
      }
    } catch (e) {
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

// ── Audio analysis via Python Flask backend ──────────────────────
const FLASK_URL = process.env.FLASK_URL || 'http://127.0.0.1:5000';

// Helper: "3:45" → 225000 ms
function durationToMs(dur) {
  if (!dur) return null;
  const parts = dur.split(':').map(Number);
  if (parts.some(isNaN)) return null;
  if (parts.length === 2) return (parts[0] * 60 + parts[1]) * 1000;
  if (parts.length === 3) return (parts[0] * 3600 + parts[1] * 60 + parts[2]) * 1000;
  return null;
}

// Full analysis: YT playlist → Deezer match → audio features + tags + description
app.get('/api/playlist/:id/analysis', async (req, res) => {
  try {
    const { id } = req.params;
    if (!id) return res.status(400).json({ error: 'No playlist ID provided' });

    // 1. Fetch YouTube playlist
    const yt = await getYouTubePlaylist(id);

    // 2. Also do Deezer preview matching (for the player) in parallel
    const tracksWithDeezer = [];
    for (const t of yt.tracks) {
      const artist = Array.isArray(t.artists) ? t.artists.join(' ') : t.artists;
      let deezerMatch = null;
      try {
        deezerMatch = await searchDeezerPreview(t.name, artist);
      } catch (_) {}
      tracksWithDeezer.push({
        name: t.name,
        artist,
        duration: t.duration,
        duration_ms: durationToMs(t.duration),
        thumbnail: t.thumbnail,
        deezer: deezerMatch,
      });
    }

    // 3. Send to Flask for audio analysis (batch)
    //    Pass the Deezer preview URL we already found so Flask
    //    can download + analyse it directly (avoids re-searching
    //    Deezer with the raw noisy YouTube title).
    const batchBody = {
      tracks: tracksWithDeezer.map(t => ({
        name: t.name,
        artist: t.artist,
        duration_ms: t.duration_ms,
        preview_url: t.deezer?.preview || null,
      })),
    };

    let analysisResults = [];
    try {
      const flaskResp = await fetch(`${FLASK_URL}/analyze-batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(batchBody),
      });
      if (flaskResp.ok) {
        const flaskData = await flaskResp.json();
        analysisResults = flaskData.results || [];
      } else {
        console.error('Flask batch failed:', flaskResp.status);
      }
    } catch (flaskErr) {
      console.error('Flask unavailable:', flaskErr.message);
    }

    // 4. Merge Deezer preview info + analysis
    const merged = tracksWithDeezer.map((t, i) => {
      const analysis = analysisResults[i] || {};
      return {
        ...t,
        audio_features: analysis.audio_features || null,
        tags: analysis.tags || null,
        description: analysis.description || null,
      };
    });

    res.json({
      playlist: { name: yt.name, totalTracks: yt.totalTracks },
      tracks: merged,
    });
  } catch (error) {
    console.error('Analysis error:', error);
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
