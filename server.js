import express from "express";
import cors from "cors";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import fs from "fs/promises";
import dotenv from "dotenv";
import ytpl from "ytpl";
import https from "https";
import http from "http";
import crypto from "crypto";

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

const DIFFRHYTHM_BASE = process.env.DIFFRHYTHM_URL || 'https://aslp-lab-diffrhythm.hf.space';
const DIFFRHYTHM_API = `${DIFFRHYTHM_BASE}/gradio_api`;
const DIFFRHYTHM_TOKEN = process.env.DIFFRHYTHM_TOKEN || null;
const DIFFRHYTHM_MODAL_URL = process.env.DIFFRHYTHM_MODAL_URL || null;

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

// Visualize song page route
app.get('/visualize_song', (req, res) => {
  res.sendFile(__dirname + '/visualize_song.html');
});

// Trippify song page route
app.get('/trippify_song', (req, res) => {
  res.sendFile(__dirname + '/trippify_song.html');
});

// Favicon route to prevent 404 errors
app.get('/favicon.ico', (req, res) => {
  res.status(204).send();
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

// --- Hugging Face generation endpoint ---
// POST /api/generate  { prompt: string }
app.post('/api/generate', async (req, res) => {
  try {
    const { prompt, options } = req.body;
    if (!prompt || typeof prompt !== 'string') return res.status(400).json({ error: 'Missing prompt in request body' });

    const HF_API_KEY = process.env.HF_API_KEY;
    const HF_MODEL = process.env.HF_MODEL || 'diffrythm/diff-rhythm';
    if (!HF_API_KEY) return res.status(500).json({ error: 'HF_API_KEY not configured in .env' });

    // Prepare request payload. Many audio models accept { inputs, parameters } or plain inputs.
    const payload = { inputs: prompt };
    if (options && typeof options === 'object') payload.parameters = options;

    // Call Hugging Face Inference API (audio output expected)
    const hfResp = await fetch(`https://api-inference.huggingface.co/models/${HF_MODEL}`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${HF_API_KEY}`,
        'Content-Type': 'application/json',
        Accept: '*/*',
      },
      body: JSON.stringify(payload),
    });

    if (!hfResp.ok) {
      const txt = await hfResp.text().catch(() => '');
      return res.status(502).json({ error: `HF inference failed: ${hfResp.status} ${txt}` });
    }

    const contentType = hfResp.headers.get('content-type') || '';
    let audioBuffer = null;

    if (contentType.includes('application/json')) {
      // Some models return JSON with base64 audio data
      const json = await hfResp.json();
      // Try common keys
      let b64 = null;
      if (json && typeof json === 'object') {
        if (json.audio) b64 = json.audio;
        else if (json.data && Array.isArray(json.data) && json.data[0]) b64 = json.data[0];
        else if (json.generated_audio) b64 = json.generated_audio;
        else if (json.sounds && Array.isArray(json.sounds) && json.sounds[0]?.data) b64 = json.sounds[0].data;
      }

      if (!b64) {
        return res.status(502).json({ error: 'HF returned JSON but no audio field found', body: json });
      }

      // b64 may be data:audio/wav;base64,XXXXX
      const m = String(b64).match(/base64,(.+)$/);
      const raw = m ? m[1] : b64;
      audioBuffer = Buffer.from(raw, 'base64');
    } else if (contentType.startsWith('audio/') || contentType === 'application/octet-stream') {
      const ab = await hfResp.arrayBuffer();
      audioBuffer = Buffer.from(ab);
    } else {
      // Unknown content-type; try to read as arrayBuffer anyway
      try {
        const ab = await hfResp.arrayBuffer();
        audioBuffer = Buffer.from(ab);
      } catch (e) {
        const txt = await hfResp.text().catch(() => '');
        return res.status(502).json({ error: `HF returned unexpected content-type: ${contentType}`, body: txt });
      }
    }

    if (!audioBuffer) return res.status(502).json({ error: 'No audio returned from HF' });

    const genDir = join(dirname(fileURLToPath(import.meta.url)), 'generated');
    await fs.mkdir(genDir, { recursive: true });
    const filename = `generated_${Date.now()}.wav`;
    const filepath = join(genDir, filename);
    await fs.writeFile(filepath, audioBuffer);

    // Return public URL
    return res.json({ url: `/generated/${filename}` });
  } catch (e) {
    console.error('Generation error:', e);
    return res.status(500).json({ error: e.message });
  }
});

// --- DiffRhythm (Hugging Face Space) generation endpoint ---
// ── GPT-4 negative style prompt (genre contrast for CFG) ────────────
async function generateNegativeStylePrompt(stylePrompt) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return null; // gracefully degrade — Modal will fall back to zeros

  try {
    const resp = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        temperature: 0.7,
        max_tokens: 60,
        messages: [
          {
            role: 'system',
            content: [
              'You are a music production expert. Given a music style description,',
              'output a SHORT (8-15 words) description of the OPPOSITE musical style.',
              'Focus on contrasting: genre, instrumentation, mood, tempo, and energy.',
              'This will be used as a negative style prompt for classifier-free guidance,',
              'so describe real musical characteristics — not negations.',
              '',
              'Examples:',
              'Input: "dark trap, 808 bass, hi-hats, aggressive"',
              'Output: "bright acoustic folk, gentle fingerpicked guitar, soft and warm"',
              '',
              'Input: "dreamy lo-fi hip hop, mellow piano, vinyl crackle"',
              'Output: "harsh industrial metal, distorted electric guitar, fast and aggressive"',
              '',
              'Input: "epic orchestral cinematic, brass, strings, powerful"',
              'Output: "minimal lo-fi bedroom pop, soft ukulele, quiet and intimate"',
              '',
              'Output ONLY the contrasting style description, nothing else.',
            ].join('\n'),
          },
          { role: 'user', content: stylePrompt },
        ],
      }),
    });

    if (!resp.ok) return null;
    const data = await resp.json();
    const result = (data.choices?.[0]?.message?.content || '').trim();
    if (result) console.log(`[negative-style] "${stylePrompt.slice(0, 60)}…" → "${result}"`);
    return result || null;
  } catch (e) {
    console.warn('[negative-style] GPT call failed, will use zero vector:', e.message);
    return null;
  }
}

// ── GPT-4 lyric generation ──────────────────────────────────────────
function clampDuration(seconds, min = 30, max = 240) {
  const n = Number(seconds);
  if (!Number.isFinite(n)) return min;
  return Math.min(max, Math.max(min, n));
}

function forceAtlantaTrapIfRap(prompt) {
  const text = String(prompt || "");
  return /\brap\b/i.test(text) ? "atlanta trap" : text;
}

function randomDuration(min = 120, max = 240) {
  const a = Math.max(1, Number(min));
  const b = Math.max(a, Number(max));
  return Math.floor(a + Math.random() * (b - a + 1));
}

async function generateLyricsWithGPT(stylePrompt, durationSec = 240) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY not configured — cannot generate lyrics');

  const dur = clampDuration(durationSec);
  const maxLines = Math.min(32, Math.max(14, Math.round(dur / 7.5)));
  const resp = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: 'gpt-4o',
      temperature: 0.85,
      max_tokens: 600,
      messages: [
        {
          role: 'system',
          content: [
            'You are a professional songwriter. Given a music style description, write original song lyrics.',
            '',
            'Rules:',
            `- Write exactly ${maxLines} lines of lyrics.`,
            '- One line of lyrics per line — no blank lines between them.',
            '- Do NOT include timestamps, section headers (Verse, Chorus, Bridge), or any formatting.',
            '- Lyrics should match the mood, genre, and energy described in the style prompt.',
            '- Write vivid, poetic, singable English lyrics — avoid generic filler.',
            '- Include a hook/chorus that repeats once or twice.',
            '- Output ONLY the raw lyric lines, nothing else — no title, no commentary, no numbering.',
          ].join('\n'),
        },
        {
          role: 'user',
          content: `Style: ${stylePrompt}`,
        },
      ],
    }),
  });

  if (!resp.ok) {
    const txt = await resp.text().catch(() => '');
    throw new Error(`OpenAI lyrics generation failed: ${resp.status} ${txt}`);
  }

  const data = await resp.json();
  const raw = (data.choices?.[0]?.message?.content || '').trim();
  if (!raw) {
    throw new Error('GPT returned empty lyrics');
  }

  // Clean: remove any accidental blank lines or section headers
  const plainLyrics = raw
    .split('\n')
    .map(l => l.trim())
    .filter(l => l && !/^\[.*\]$/.test(l) && !/^(verse|chorus|bridge|hook|outro|intro|pre-chorus)\s*\d*\s*:?\s*$/i.test(l))
    .join('\n');

  if (!plainLyrics) {
    throw new Error('GPT returned no usable lyric lines');
  }

  // Convert plain lyrics into properly timestamped LRC using the existing helper
  const lrc = generateLrcFromText(plainLyrics, durationSec, maxLines);
  console.log('[lyrics-gen] Plain lyrics:\n' + plainLyrics);
  console.log('[lyrics-gen] LRC:\n' + lrc);
  return { lrc, plainLyrics };
}

// POST /api/diffrhythm { prompt: string, lyrics?: string, options?: { duration, steps, cfg_strength, seed, randomize_seed, file_type, odeint_method, preference, withLyrics } }
function formatLrcTimestamp(totalSeconds) {
  const mm = Math.floor(totalSeconds / 60);
  const ss = Math.floor(totalSeconds % 60);
  const cc = Math.floor((totalSeconds - Math.floor(totalSeconds)) * 100);
  return `[${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}.${String(cc).padStart(2, '0')}]`;
}

function generateLrcFromText(text, durationSec = 240, maxLines = 24) {
  const clean = String(text || '')
    .replace(/\r/g, '\n')
    .split(/[\n.!?]+/g)
    .map(s => s.trim())
    .filter(Boolean);

  const lines = (clean.length ? clean : ['Instrumental']).slice(0, maxLines);
  const startOffset = 4.0;
  const endTime = Math.max(startOffset + 5, Math.min(clampDuration(durationSec), 240) - 2);
  const step = (endTime - startOffset) / Math.max(lines.length, 1);

  return lines
    .map((line, i) => `${formatLrcTimestamp(startOffset + step * i)} ${line}`)
    .join('\n');
}

async function waitForGradioComplete(url, timeoutMs = 15 * 60 * 1000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const res = await fetch(url, {
    headers: {
      Accept: 'text/event-stream',
      ...(DIFFRHYTHM_TOKEN ? { Authorization: `Bearer ${DIFFRHYTHM_TOKEN}` } : {}),
    },
    signal: controller.signal,
  });
  if (!res.ok) {
    clearTimeout(timer);
    throw new Error(`DiffRhythm stream error: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let lastEvent = null;
  let lastData = null;

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf('\n\n')) >= 0) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);

        let eventType = null;
        let dataStr = '';
        for (const line of chunk.split('\n')) {
          if (line.startsWith('event:')) eventType = line.slice(6).trim();
          if (line.startsWith('data:')) dataStr += line.slice(5).trim();
        }

        if (eventType) lastEvent = eventType;
        if (dataStr !== undefined) lastData = dataStr;

        if (eventType === 'error') {
          const detail = dataStr && dataStr !== 'null' ? dataStr : 'null';
          throw new Error(`DiffRhythm error (event: error, data: ${detail})`);
        }

        if (eventType === 'complete') {
          if (!dataStr || dataStr === 'null') return null;
          try {
            return JSON.parse(dataStr);
          } catch {
            return null;
          }
        }
      }
    }
  } finally {
    clearTimeout(timer);
  }

  const suffix = lastEvent ? ` (last event: ${lastEvent}${lastData ? `, data: ${lastData}` : ''})` : '';
  throw new Error('DiffRhythm stream ended without completion' + suffix);
}

async function downloadToGenerated(url, ext = 'mp3') {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to download DiffRhythm audio: ${resp.status}`);
  const ab = await resp.arrayBuffer();
  const genDir = join(dirname(fileURLToPath(import.meta.url)), 'generated');
  await fs.mkdir(genDir, { recursive: true });
  const suffix = crypto.randomBytes(6).toString('hex');
  const filename = `diffrhythm_${Date.now()}_${suffix}.${ext}`;
  const filepath = join(genDir, filename);
  await fs.writeFile(filepath, Buffer.from(ab));
  return `/generated/${filename}`;
}

async function saveBase64ToGenerated(b64, ext = 'mp3') {
  const genDir = join(dirname(fileURLToPath(import.meta.url)), 'generated');
  await fs.mkdir(genDir, { recursive: true });
  const suffix = crypto.randomBytes(6).toString('hex');
  const filename = `diffrhythm_${Date.now()}_${suffix}.${ext}`;
  const filepath = join(genDir, filename);
  const buf = Buffer.from(b64, 'base64');
  await fs.writeFile(filepath, buf);
  return `/generated/${filename}`;
}

async function callDiffRhythm(prompt, lrc, options = {}) {
  if (DIFFRHYTHM_MODAL_URL) {
    console.log('[callDiffRhythm] Sending to Modal:', {
      text_prompt: prompt?.slice(0, 80) + '…',
      lyrics_length: lrc?.length,
      lyrics_preview: lrc?.slice(0, 120),
      vocal_flag: !!options.vocal_flag,
      negative_text_prompt: options.negative_text_prompt || null,
    });
    const resp = await fetch(DIFFRHYTHM_MODAL_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text_prompt: prompt,
        lyrics: lrc,
        options,
        vocal_flag: !!options.vocal_flag,
        negative_text_prompt: options.negative_text_prompt || null,
      }),
    });
    if (!resp.ok) {
      const txt = await resp.text().catch(() => '');
      throw new Error(`Modal DiffRhythm failed: ${resp.status} ${txt}`);
    }
    const data = await resp.json();
    if (data.error) throw new Error(`Modal DiffRhythm error: ${data.error}`);
    if (!data.audio_base64) throw new Error('Modal DiffRhythm returned no audio_base64');
    const ext = data.file_type || options.file_type || 'mp3';
    const url = await saveBase64ToGenerated(data.audio_base64, ext);
    return { url, lyrics_used: lrc };
  }

  const callResp = await fetch(`${DIFFRHYTHM_API}/call/infer_music`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(DIFFRHYTHM_TOKEN ? { Authorization: `Bearer ${DIFFRHYTHM_TOKEN}` } : {}),
    },
    body: JSON.stringify({
      data: [
        lrc,
        null,
        prompt,
        'text',
        Number.isFinite(options.seed) ? Number(options.seed) : 0,
        options.randomize_seed !== false,
        Number(options.steps || 50),
        Number(options.cfg_strength || 4.0),
        options.file_type || 'mp3',
        options.odeint_method || 'euler',
        options.preference || 'speed first',
        Number.isFinite(Number(options.duration))
          ? clampDuration(options.duration)
          : randomDuration(120, 240),
      ],
    }),
  });

  if (!callResp.ok) {
    const txt = await callResp.text().catch(() => '');
    throw new Error(`DiffRhythm call failed: ${callResp.status} ${txt}`);
  }

  const { event_id } = await callResp.json();
  if (!event_id) throw new Error('DiffRhythm did not return event_id');

  const result = await waitForGradioComplete(`${DIFFRHYTHM_API}/call/infer_music/${event_id}`);
  const file = result?.data?.[0] || null;
  const fileUrl = file?.url || null;

  let url = fileUrl;
  if (fileUrl) {
    try {
      url = await downloadToGenerated(fileUrl, options.file_type || 'mp3');
    } catch (_) {
      url = fileUrl;
    }
  }

  if (!url) throw new Error('DiffRhythm returned no audio URL');
  return { url, lyrics_used: lrc };
}

app.post('/api/diffrhythm', async (req, res) => {
  try {
    const { prompt, lyrics, options } = req.body || {};
    if (!prompt || typeof prompt !== 'string') {
      return res.status(400).json({ error: 'Missing prompt in request body' });
    }

    const promptForLyrics = forceAtlantaTrapIfRap(prompt);
    const opts = options && typeof options === 'object' ? options : {};
    const duration = Number.isFinite(Number(opts.duration))
      ? clampDuration(opts.duration)
      : randomDuration(120, 240);
    const steps = Number(opts.steps || 50);
    const cfgStrength = Number(opts.cfg_strength || 4.0);
    const seed = Number.isFinite(opts.seed) ? Number(opts.seed) : 0;
    const randomizeSeed = opts.randomize_seed !== false;
    const fileType = opts.file_type || 'mp3';
    const odeintMethod = opts.odeint_method || 'euler';
    const preference = opts.preference || 'speed first';

    const withLyrics = !!opts.withLyrics;

    let lrc;
    let plainLyrics = null;
    let negativeStylePrompt = null;
    if (lyrics && typeof lyrics === 'string' && lyrics.trim()) {
      lrc = lyrics.trim();
      // Still generate a negative prompt for genre steering
      negativeStylePrompt = await generateNegativeStylePrompt(promptForLyrics);
    } else if (withLyrics) {
      console.log('[diffrhythm] Generating lyrics + negative style prompt via GPT…');
      // Run both GPT calls in parallel for speed
      const [generated, negPrompt] = await Promise.all([
        generateLyricsWithGPT(promptForLyrics, duration),
        generateNegativeStylePrompt(promptForLyrics),
      ]);
      lrc = generated.lrc;
      plainLyrics = generated.plainLyrics;
      negativeStylePrompt = negPrompt;
    } else {
      lrc = '[00:00.00] Instrumental';
    }

    const result = await callDiffRhythm(promptForLyrics, lrc, {
      duration,
      steps,
      cfg_strength: cfgStrength,
      seed,
      randomize_seed: randomizeSeed,
      file_type: fileType,
      odeint_method: odeintMethod,
      preference,
      lrc_max_lines: opts.lrc_max_lines || 24,
      vocal_flag: withLyrics,
      negative_text_prompt: negativeStylePrompt,
    });

    result.lyrics_used = plainLyrics || lrc;
    return res.json(result);
  } catch (e) {
    console.error('DiffRhythm error:', e);
    return res.status(500).json({ error: e.message });
  }
});

// --- Full pipeline: analyze playlist -> use summary -> DiffRhythm ---
// POST /api/playlist/:id/generate-song { options?: { duration, steps, cfg_strength, seed, randomize_seed, file_type, odeint_method, preference } }
app.post('/api/playlist/:id/generate-song', async (req, res) => {
  try {
    const { id } = req.params;
    if (!id) return res.status(400).json({ error: 'No playlist ID provided' });

    // Reuse analysis pipeline to produce playlist_summary
    // This mirrors GET /api/playlist/:id/analysis but in POST flow
    const yt = await getYouTubePlaylist(id);

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

    // Sample up to 10 random tracks (with previews) for faster analysis
    const withPreviews = tracksWithDeezer.filter(t => t.deezer?.preview);
    const sampled = withPreviews.length <= 10
      ? withPreviews
      : withPreviews.sort(() => Math.random() - 0.5).slice(0, 10);
    console.log(`[generate] Sampling ${sampled.length}/${tracksWithDeezer.length} tracks for analysis`);

    const batchBody = {
      tracks: sampled.map(t => ({
        name: t.name,
        artist: t.artist,
        duration_ms: t.duration_ms,
        preview_url: t.deezer?.preview || null,
      })),
    };

    let analysisResults = [];
    let playlistSummary = null;
    let generatedLyrics = null;

    const backendUrl = MODAL_URL || `${FLASK_URL}/analyze-batch`;
    const backendName = MODAL_URL ? 'Modal' : 'Flask';
    console.log(`[generate] Using ${backendName} backend: ${backendUrl}`);

    try {
      const resp = await fetch(backendUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(batchBody),
      });
      if (resp.ok) {
        const data = await resp.json();
        analysisResults = data.results || [];
        playlistSummary = data.playlist_summary || null;
        generatedLyrics = data.generated_lyrics || null;
      } else {
        console.error(`${backendName} batch failed:`, resp.status);
      }
    } catch (err) {
      console.error(`${backendName} unavailable:`, err.message);
    }

    const merged = tracksWithDeezer.map((t, i) => {
      const analysis = analysisResults[i] || {};
      return {
        ...t,
        audio_features: analysis.audio_features || null,
        tags: analysis.tags || null,
        description: analysis.description || null,
      };
    });

    if (!playlistSummary) {
      return res.status(502).json({ error: 'No playlist summary returned from analysis backend', tracks: merged, playlist: { name: yt.name, totalTracks: yt.totalTracks } });
    }

    // Call ACE-Step using the summary as style prompt and generated lyrics
    const { options } = req.body || {};
    const lyrics = (options?.lyrics && typeof options.lyrics === 'string' && options.lyrics.trim())
      ? options.lyrics.trim()
      : (generatedLyrics || '');
    let drResult;
    try {
      drResult = await callDiffRhythm(playlistSummary, lyrics, {
        duration: Number.isFinite(Number(options?.duration))
          ? clampDuration(options.duration)
          : randomDuration(120, 240),
        seed: Number.isFinite(options?.seed) ? Number(options.seed) : 1,
        file_type: options?.file_type || 'mp3',
      });
    } catch (err) {
      return res.status(502).json({ error: err.message, tracks: merged, playlist: { name: yt.name, totalTracks: yt.totalTracks }, playlist_summary: playlistSummary, generated_lyrics: generatedLyrics });
    }

    res.json({
      playlist: { name: yt.name, totalTracks: yt.totalTracks },
      tracks: merged,
      playlist_summary: playlistSummary,
      generated_lyrics: generatedLyrics,
      generated: { url: drResult.url },
    });
  } catch (e) {
    console.error('Generate-song error:', e);
    return res.status(500).json({ error: e.message });
  }
});

// --- Deezer preview support ---

// Parse a YouTube video title into { trackName, artists[] }
function parseYouTubeTitle(rawTitle, channelName) {
  let title = rawTitle || "";

  // Remove parenthetical/bracketed noise: (Official Video), [HD], etc.
  title = title.replace(/\([^)]*\)/g, "").replace(/\[[^\]]*\]/g, "");

  // Remove pipe-separated suffixes like "| EL ÚLTIMO TOUR DEL MUNDO"
  title = title.replace(/\|.*$/, "");

  // Remove common YouTube noise words (including K-pop M/V)
  title = title.replace(/official\s*video|official\s*music\s*video|official\s*audio|music\s*video|lyric\s*video|official|lyrics|audio|video|HD|HQ|4K|visualizer|\bM\/V\b|\bMV\b/gi, "");

  // Remove credit tags like "🎥By. Ryan Lynch", "Prod. By ..."
  title = title.replace(/🎥.*$/g, "");
  title = title.replace(/\bprod\.?\s*by\.?\s.*/gi, "");
  title = title.replace(/\bdir\.?\s*by\.?\s.*/gi, "");

  // Extract quoted track name before stripping quotes
  // Handles K-pop pattern: TWICE "FANCY" M/V → track = FANCY, artist = TWICE
  let quotedTrack = null;
  const quoteMatch = title.match(/["""]([^"""]+)["""]/);
  if (quoteMatch) {
    quotedTrack = quoteMatch[1].trim();
  }

  // Remove stray quotes
  title = title.replace(/["""]/g, "");

  title = title.replace(/\s+/g, " ").trim();

  // Split on " - " or " – " to separate artist(s) from track name
  let trackName = title;
  let parsedArtists = [];

  // If we found a quoted track name (e.g. TWICE "FANCY"), use it directly
  if (quotedTrack) {
    trackName = quotedTrack;
    // Everything outside the quotes is the artist(s) — strip stray dashes/punctuation
    const artistPart = title.replace(quotedTrack, "").replace(/[-–—]+/g, " ").replace(/\s+/g, " ").trim();
    if (artistPart) {
      parsedArtists = artistPart.split(/\s+x\s+|\s*&\s*|\s*,\s*|\s+feat\.?\s+|\s+ft\.?\s+|\s+featuring\s+/i).map(a => a.trim()).filter(Boolean);
    }
  } else {
    const dashMatch = title.match(/^(.+?)\s*[-–]\s+(.+)$/);
    if (dashMatch) {
      const left = dashMatch[1].trim();   // artist side
      const right = dashMatch[2].trim();  // track side

      // Split artists on " x ", " X ", " & ", ", ", "feat.", "ft.", "featuring"
      parsedArtists = left.split(/\s+x\s+|\s*&\s*|\s*,\s*|\s+feat\.?\s+|\s+ft\.?\s+|\s+featuring\s+/i).map(a => a.trim()).filter(Boolean);
      trackName = right;
    }
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
          cover: bestResult.album?.cover_medium || bestResult.album?.cover || null,
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
          thumbnail: t.thumbnail || null,
          deezer: found,
        });
      } catch (e) {
        out.push({ name: title, artists: artist, duration: t.duration, thumbnail: t.thumbnail || null, deezer: null, error: e.message });
      }
    }

    res.json({ playlist: { name: yt.name, totalTracks: yt.totalTracks }, tracks: out });
  } catch (error) {
    console.error('Deezer preview error:', error);
    res.status(500).json({ error: error.message });
  }
});

// ── Audio analysis backend (Modal GPU or Flask fallback) ─────────
const MODAL_URL = process.env.MODAL_URL || null;
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

// Full analysis: YT playlist → Deezer match → Modal/Flask analysis + GPT-4 summary
app.get('/api/playlist/:id/analysis', async (req, res) => {
  try {
    const { id } = req.params;
    if (!id) return res.status(400).json({ error: 'No playlist ID provided' });

    // 1. Fetch YouTube playlist
    const yt = await getYouTubePlaylist(id);

    // 2. Deezer preview matching
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

    // 3. Sample up to 10 random tracks (with previews) for faster analysis
    const withPreviews = tracksWithDeezer
      .map((t, i) => ({ ...t, _origIdx: i }))
      .filter(t => t.deezer?.preview);
    const sampled = withPreviews.length <= 10
      ? withPreviews
      : withPreviews.sort(() => Math.random() - 0.5).slice(0, 10);
    console.log(`[analysis] Sampling ${sampled.length}/${tracksWithDeezer.length} tracks for analysis`);

    const batchBody = {
      tracks: sampled.map(t => ({
        name: t.name,
        artist: t.artist,
        duration_ms: t.duration_ms,
        preview_url: t.deezer?.preview || null,
      })),
    };

    let analysisResults = [];
    let playlistSummary = null;

    const backendUrl = MODAL_URL || `${FLASK_URL}/analyze-batch`;
    const backendName = MODAL_URL ? 'Modal' : 'Flask';
    console.log(`[analysis] Using ${backendName} backend: ${backendUrl}`);

    try {
      const resp = await fetch(backendUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(batchBody),
      });
      if (resp.ok) {
        const data = await resp.json();
        analysisResults = data.results || [];
        playlistSummary = data.playlist_summary || null;
      } else {
        console.error(`${backendName} batch failed:`, resp.status);
      }
    } catch (err) {
      console.error(`${backendName} unavailable:`, err.message);
    }

    // 4. Merge analysis back into sampled tracks, leave others without analysis
    const analysisMap = new Map();
    sampled.forEach((t, i) => { analysisMap.set(t._origIdx, analysisResults[i] || {}); });
    const merged = tracksWithDeezer.map((t, i) => {
      const analysis = analysisMap.get(i) || {};
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
      playlist_summary: playlistSummary,
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
