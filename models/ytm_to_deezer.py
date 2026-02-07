"""
ytm_to_deezer.py
~~~~~~~~~~~~~~~~
Pipeline that bridges YouTube Music playlists to Deezer previews for
audio-feature extraction.

Stages
------
1. Fetch playlist tracks from YouTube Music  (ytmusicapi).
2. Match each track on Deezer via its search API.
3. Download Deezer 30-second MP3 preview (cached locally).
4. Extract rich audio features with librosa (tempo, key, mode,
   spectral shape, dynamics, rhythm, timbre, MFCCs, chroma).
5. Auto-tag with PANNs (genre, instruments, vocals) on GPU.
6. Produce a detailed natural-language description per track,
   designed to be refined by an LLM into a music-generation prompt.
"""

import os
import re
import json
import time
import hashlib
from typing import Optional, Dict, Any, List

import requests
import numpy as np
import librosa
import torch
from ytmusicapi import YTMusic

__all__ = [
    "extract_audio_features",
    "extract_tags",
    "describe_track",
    "fetch_ytm_playlist_tracks",
    "try_deezer_preview",
    "process_ytm_playlist",
]

# Pitch-class names for key detection
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ----------------------------
# Config
# ----------------------------
HEADERS_PATH = "headers_auth.json"     # produced by `ytmusicapi browserauth`
CACHE_DIR = "cache_deezer_previews"
OUT_DIR = "out_features"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

DEEZER_SEARCH_URL = "https://api.deezer.com/search"
DEEZER_TRACK_URL = "https://api.deezer.com/track"

# ----------------------------
# Small utilities
# ----------------------------

def norm(s: str) -> str:
    s = (s or "").lower()
    # remove bracketed "remaster" / "live" / etc
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s

def parse_duration_to_ms(duration: Optional[str]) -> Optional[int]:
    """
    YouTube Music often gives duration like "3:45" or "1:02:10".
    """
    if not duration:
        return None
    parts = duration.strip().split(":")
    if not all(p.isdigit() for p in parts):
        return None
    parts = [int(p) for p in parts]
    if len(parts) == 2:
        m, s = parts
        return (m * 60 + s) * 1000
    if len(parts) == 3:
        h, m, s = parts
        return (h * 3600 + m * 60 + s) * 1000
    return None

def download_file(url: str, out_path: str, timeout: int = 30) -> str:
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return out_path
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

def extract_audio_features(audio_path: str, sr: int = 22050) -> Dict[str, Any]:
    """
    Extract a rich set of audio features from a preview clip.

    The output is designed to give an LLM enough detail to write a
    text-conditioned music-generation prompt.  All values are JSON-safe
    Python floats/lists/strings.

    Feature groups
    --------------
    tempo_and_rhythm   – BPM, beat regularity, onset stats
    key_and_tonality   – estimated key, mode (major/minor), mode confidence
    dynamics            – RMS mean/std/max, dynamic range dB, loudness profile
    spectral_shape      – centroid, bandwidth, rolloff, flatness, contrast
    timbre              – MFCCs (mean of first 13 coefficients)
    chroma              – mean chroma vector (12 pitch classes)
    texture             – zero-crossing rate, harmonic ratio
    """
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    duration_sec = float(len(y) / sr)

    # ── Tempo & rhythm ──────────────────────────────────────────
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo).flat[0])

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beat_times) > 1:
        ibis = np.diff(beat_times)  # inter-beat intervals
        beat_regularity = float(1.0 - np.std(ibis) / (np.mean(ibis) + 1e-8))
    else:
        beat_regularity = 0.0

    onset = librosa.onset.onset_strength(y=y, sr=sr)

    # ── Key & tonality ──────────────────────────────────────────
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma_cq, axis=1)  # (12,)

    # Simple key estimation: strongest pitch class
    key_idx = int(np.argmax(chroma_mean))
    key_name = _PITCH_CLASSES[key_idx]

    # Major vs minor heuristic via Krumhansl profiles
    _major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    _minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    corr_major = float(np.corrcoef(np.roll(chroma_mean, -key_idx), _major)[0, 1])
    corr_minor = float(np.corrcoef(np.roll(chroma_mean, -key_idx), _minor)[0, 1])
    mode = "major" if corr_major >= corr_minor else "minor"
    mode_confidence = round(float(max(corr_major, corr_minor)), 3)

    # ── Dynamics ────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_max = float(np.max(rms))
    rms_std = float(np.std(rms))
    # Dynamic range in dB (difference between loudest and quietest frames)
    rms_floor = np.maximum(rms, 1e-10)
    dynamic_range_db = float(20.0 * np.log10(np.max(rms_floor) / np.min(rms_floor)))

    # Temporal energy profile: divide into 4 quarters
    quarter = len(rms) // 4 or 1
    energy_profile = [float(np.mean(rms[i * quarter:(i + 1) * quarter])) for i in range(4)]

    # ── Spectral shape ──────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # (7, T)

    # ── Timbre (MFCCs) ──────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = [round(float(c), 4) for c in np.mean(mfcc, axis=1)]
    mfcc_std = [round(float(c), 4) for c in np.std(mfcc, axis=1)]

    # ── Texture ─────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    y_harm, y_perc = librosa.effects.hpss(y)
    harmonic_ratio = float(np.sum(y_harm ** 2) / (np.sum(y ** 2) + 1e-10))

    return {
        # Tempo & rhythm
        "tempo_bpm": tempo,
        "beat_regularity": round(beat_regularity, 4),
        "onset_strength_mean": round(float(np.mean(onset)), 4),
        "onset_strength_std": round(float(np.std(onset)), 4),
        "duration_sec": round(duration_sec, 2),
        # Key & tonality
        "estimated_key": key_name,
        "mode": mode,
        "mode_confidence": mode_confidence,
        "chroma_mean": [round(float(c), 4) for c in chroma_mean],
        # Dynamics
        "rms_mean": round(rms_mean, 5),
        "rms_std": round(rms_std, 5),
        "rms_max": round(rms_max, 5),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "energy_profile_quarters": [round(e, 5) for e in energy_profile],
        # Spectral shape
        "spectral_centroid_mean": round(float(np.mean(centroid)), 2),
        "spectral_centroid_std": round(float(np.std(centroid)), 2),
        "spectral_bandwidth_mean": round(float(np.mean(bandwidth)), 2),
        "spectral_rolloff_mean": round(float(np.mean(rolloff)), 2),
        "spectral_flatness_mean": round(float(np.mean(flatness)), 6),
        "spectral_contrast_mean": [round(float(c), 4) for c in np.mean(contrast, axis=1)],
        # Timbre
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
        # Texture
        "zero_crossing_rate_mean": round(float(np.mean(zcr)), 6),
        "harmonic_ratio": round(harmonic_ratio, 4),
    }


# Keep backward-compatible alias
extract_basic_audio_features = extract_audio_features

# ----------------------------
# PANNs music tagging (GPU)
# ----------------------------

# Music-relevant AudioSet label indices (genre, instrument, vocal style)
MUSIC_LABELS: Dict[str, List[int]] = {
    "genre": [
        216,  # Pop music
        217,  # Hip hop music
        219,  # Rock music
        220,  # Heavy metal
        221,  # Punk rock
        223,  # Progressive rock
        224,  # Rock and roll
        225,  # Psychedelic rock
        226,  # Rhythm and blues
        227,  # Soul music
        228,  # Reggae
        229,  # Country
        233,  # Folk music
        235,  # Jazz
        237,  # Classical music
        239,  # Electronic music
        243,  # Drum and bass
        244,  # Electronica
        245,  # Electronic dance music
        251,  # Blues
        254,  # Vocal music
    ],
    "instrument": [
        140,  # Guitar
        141,  # Electric guitar
        142,  # Bass guitar
        143,  # Acoustic guitar
        153,  # Piano
        154,  # Electric piano
        156,  # Electronic organ
        162,  # Drum kit
        163,  # Drum machine
        164,  # Drum
        165,  # Snare drum
        168,  # Bass drum
        191,  # Violin, fiddle
        194,  # Double bass
        196,  # Flute
    ],
    "vocal": [
        27,   # Singing
        32,   # Male singing
        33,   # Female singing
        34,   # Child singing
        35,   # Synthetic singing
    ],
}

# Lazy-loaded singleton tagger so we only load the model once
_panns_tagger = None
_panns_labels: List[str] = []

def _get_tagger():
    """Return the PANNs AudioTagging model (loaded once, on GPU if available)."""
    global _panns_tagger, _panns_labels
    if _panns_tagger is None:
        import csv
        from panns_inference import AudioTagging

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _panns_tagger = AudioTagging(checkpoint_path=None, device=device)

        # Load label names
        labels_csv = os.path.join(
            os.path.expanduser("~"), "panns_data", "class_labels_indices.csv"
        )
        with open(labels_csv) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            _panns_labels = [row[2] for row in reader]

    return _panns_tagger, _panns_labels


# Per-category thresholds — vocals are under-represented in 30-second
# preview clips (intros, instrumental breaks), so we use a much lower
# threshold there to avoid falsely labelling everything "instrumental".
_CATEGORY_THRESHOLDS: Dict[str, float] = {
    "genre": 0.03,
    "instrument": 0.03,
    "vocal": 0.01,          # very low — we'd rather over-report than miss
}

# Genres that almost always contain vocals.  Used by describe_track()
# to avoid the misleading "Likely instrumental" label.
_VOCAL_GENRES = {
    "Pop music", "Hip hop music", "Rock music", "Heavy metal",
    "Punk rock", "Rock and roll", "Rhythm and blues", "Soul music",
    "Reggae", "Country", "Folk music", "Blues", "Vocal music",
}


def extract_tags(
    audio_path: str,
    top_n: int = 5,
    vocal_window_sec: float = 10.0,
    vocal_hop_sec: float = 5.0,
) -> Dict[str, Any]:
    """
    Auto-tag an audio file using PANNs (Pretrained Audio Neural Networks).

    Genre and instrument tags are extracted from the **full clip** (best
    with more context).  Vocal tags use a **sliding-window scan** — the
    clip is split into overlapping windows and the *best* vocal
    probabilities across all windows are kept.  This avoids the common
    problem of missing vocals that only appear later in the 30-second
    Deezer preview (e.g. after an intro or instrumental break).

    Parameters
    ----------
    audio_path : str
        Path to an audio file (MP3, WAV, etc.).
    top_n : int
        Max tags to return per category.
    vocal_window_sec : float
        Window length in seconds for the vocal scan (default 10 s).
    vocal_hop_sec : float
        Hop between consecutive vocal windows (default 5 s → 50 % overlap).

    Returns
    -------
    dict with keys "genre", "instrument", "vocal", each mapping to a list
    of {"label": str, "prob": float} sorted by descending probability.
    Also includes "_vocal_prob_max" (float) — the highest raw vocal
    probability across all windows.
    """
    tagger, labels = _get_tagger()

    SR_PANNS = 32_000  # PANNs expects 32 kHz
    y, _ = librosa.load(audio_path, sr=SR_PANNS, mono=True)

    # ── Full-clip inference for genre & instrument ──────────────
    waveform = y[np.newaxis, :]  # (1, samples)
    with torch.no_grad():
        clipwise_output, _ = tagger.inference(waveform)
    full_probs = clipwise_output[0]  # (527,)

    # ── Windowed vocal scan ─────────────────────────────────────
    win_samples = int(vocal_window_sec * SR_PANNS)
    hop_samples = int(vocal_hop_sec * SR_PANNS)

    # Build windows; fall back to full clip if it's shorter than one window
    starts = list(range(0, max(1, len(y) - win_samples + 1), hop_samples))
    if not starts:
        starts = [0]

    vocal_indices = MUSIC_LABELS["vocal"]
    # Track the best probability seen per vocal label across all windows
    best_vocal: Dict[int, float] = {idx: 0.0 for idx in vocal_indices}

    for s in starts:
        chunk = y[s : s + win_samples]
        if len(chunk) < SR_PANNS:  # skip tiny tail (<1 s)
            continue
        wav = chunk[np.newaxis, :]
        with torch.no_grad():
            co, _ = tagger.inference(wav)
        p = co[0]
        for idx in vocal_indices:
            best_vocal[idx] = max(best_vocal[idx], float(p[idx]))

    # ── Assemble results ────────────────────────────────────────
    result: Dict[str, Any] = {}
    for category, indices in MUSIC_LABELS.items():
        thresh = _CATEGORY_THRESHOLDS.get(category, 0.03)
        cat_tags = []
        for idx in indices:
            if category == "vocal":
                p = best_vocal.get(idx, 0.0)   # use windowed max
            else:
                p = float(full_probs[idx])      # use full-clip
            if p >= thresh:
                cat_tags.append({"label": labels[idx], "prob": round(p, 4)})
        cat_tags.sort(key=lambda x: x["prob"], reverse=True)
        result[category] = cat_tags[:top_n]

    # Expose peak vocal prob for downstream reasoning
    result["_vocal_prob_max"] = round(
        max(best_vocal.values()) if best_vocal else 0.0, 4
    )

    return result


# ----------------------------
# YouTube Music metadata
# ----------------------------

def fetch_ytm_playlist_tracks(playlist_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Returns normalized track dicts:
      { name, artist, duration_ms, ytm_videoId, ytm_playlistItemId }
    """
    ytm = YTMusic(HEADERS_PATH)
    pl = ytm.get_playlist(playlist_id, limit=limit)
    tracks = pl.get("tracks", []) or []

    out = []
    for t in tracks:
        title = t.get("title") or ""
        artists = t.get("artists") or []
        artist_name = artists[0].get("name") if artists else ""
        duration_ms = parse_duration_to_ms(t.get("duration"))
        out.append({
            "name": title,
            "artist": artist_name,
            "duration_ms": duration_ms,
            "ytm_videoId": t.get("videoId"),
            "ytm_playlistItemId": t.get("playlistItemId"),
        })
    return out

# ----------------------------
# Deezer matching + preview
# ----------------------------

def deezer_search(track: str, artist: str, limit: int = 10) -> Dict[str, Any]:
    # Deezer supports advanced search query strings
    q = f'artist:"{artist}" track:"{track}"'
    resp = requests.get(DEEZER_SEARCH_URL, params={"q": q, "limit": limit}, timeout=20)
    resp.raise_for_status()
    return resp.json()

def deezer_track(track_id: int) -> Dict[str, Any]:
    resp = requests.get(f"{DEEZER_TRACK_URL}/{track_id}", timeout=20)
    resp.raise_for_status()
    return resp.json()

def pick_best_deezer_hit(
    hits: List[Dict[str, Any]],
    track_name: str,
    artist_name: str,
    duration_ms: Optional[int],
) -> Optional[Dict[str, Any]]:
    """
    Simple heuristic scoring:
      - title match
      - artist match
      - duration closeness (optional)
      - prefers preview availability
    """
    tn = norm(track_name)
    an = norm(artist_name)
    target_sec = int(duration_ms / 1000) if duration_ms else None

    best = None
    for h in hits:
        title = norm(h.get("title", ""))
        artist = norm((h.get("artist") or {}).get("name", ""))
        dur_sec = h.get("duration")  # seconds

        score = 0.0
        if title == tn:
            score += 3.0
        elif tn and (tn in title or title in tn):
            score += 1.5

        if artist == an:
            score += 3.0
        elif an and (an in artist or artist in an):
            score += 1.5

        if target_sec and isinstance(dur_sec, int):
            diff = abs(dur_sec - target_sec)
            score += max(0.0, 2.0 - diff / 10.0)  # within ~20 sec helps

        if h.get("preview"):
            score += 0.5

        if best is None or score > best["score"]:
            best = {"score": score, "hit": h}

    return best["hit"] if best else None

def try_deezer_preview(track_name: str, artist_name: str, duration_ms: Optional[int]) -> Dict[str, Any]:
    """
    Attempts to find and download Deezer preview.
    Returns dict with:
      {found, deezer, preview_url, audio_path, error}
    """
    result: Dict[str, Any] = {"found": False}

    data = deezer_search(track_name, artist_name, limit=10)
    hits = data.get("data", []) or []
    best = pick_best_deezer_hit(hits, track_name, artist_name, duration_ms)
    if not best or not best.get("id"):
        return result

    dz_id = int(best["id"])
    dz_full = deezer_track(dz_id)
    preview_url = dz_full.get("preview")

    result["deezer"] = {
        "id": dz_id,
        "title": dz_full.get("title"),
        "artist": (dz_full.get("artist") or {}).get("name"),
        "link": dz_full.get("link"),
        "duration_sec": dz_full.get("duration"),
        "isrc": dz_full.get("isrc"),
    }

    if not preview_url:
        return result

    try:
        h = hashlib.sha1(preview_url.encode("utf-8")).hexdigest()[:12]
        audio_path = os.path.join(CACHE_DIR, f"dz_{dz_id}_{h}.mp3")
        download_file(preview_url, audio_path)
        result.update({
            "found": True,
            "preview_url": preview_url,
            "audio_path": audio_path,
        })
    except Exception as e:
        result["error"] = str(e)

    return result

# ----------------------------
# End-to-end pipeline
# ----------------------------

def _tempo_descriptor(bpm: float) -> str:
    """Convert BPM into a human tempo marking."""
    if bpm < 40:
        return "very slow (grave)"
    if bpm < 66:
        return "slow (largo/adagio)"
    if bpm < 76:
        return "walking pace (andante)"
    if bpm < 108:
        return "moderate (moderato)"
    if bpm < 120:
        return "lively (allegretto)"
    if bpm < 156:
        return "fast (allegro)"
    if bpm < 176:
        return "very fast (vivace)"
    return "extremely fast (presto)"


def _dynamics_descriptor(rms_mean: float, dynamic_range_db: float) -> str:
    """Describe dynamics: overall loudness + dynamic range."""
    parts = []
    if rms_mean > 0.12:
        parts.append("very loud and compressed")
    elif rms_mean > 0.08:
        parts.append("loud and punchy")
    elif rms_mean > 0.04:
        parts.append("moderate loudness")
    elif rms_mean > 0.015:
        parts.append("soft and gentle")
    else:
        parts.append("very quiet and delicate")

    if dynamic_range_db > 30:
        parts.append("with a wide dynamic range (lots of crescendos and variation)")
    elif dynamic_range_db > 15:
        parts.append("with moderate dynamic variation")
    else:
        parts.append("with a narrow dynamic range (consistent loudness)")
    return ", ".join(parts)


def _spectral_descriptor(centroid: float, bandwidth: float, rolloff: float, flatness: float) -> str:
    """Describe tonal color from spectral statistics."""
    parts = []
    # brightness
    if centroid > 4000:
        parts.append("very bright, airy timbre")
    elif centroid > 3000:
        parts.append("bright timbre")
    elif centroid > 2000:
        parts.append("balanced, mid-range timbre")
    elif centroid > 1200:
        parts.append("warm, slightly dark timbre")
    else:
        parts.append("dark, bass-heavy timbre")

    # width
    if bandwidth > 2500:
        parts.append("spectrally wide and rich")
    elif bandwidth < 1200:
        parts.append("spectrally narrow and focused")

    # noisiness
    if flatness > 0.1:
        parts.append("noisy/textured sound")
    elif flatness < 0.005:
        parts.append("clean, tonal sound")

    return ", ".join(parts)


def _energy_arc(quarters: list) -> str:
    """Describe how the energy evolves across the clip."""
    if not quarters or len(quarters) < 4:
        return "steady energy"
    q = quarters
    rising = q[3] > q[0] * 1.3
    falling = q[3] < q[0] * 0.7
    middle_peak = max(q[1], q[2]) > max(q[0], q[3]) * 1.3

    if rising:
        return "energy builds throughout the clip"
    if falling:
        return "energy fades / decays over time"
    if middle_peak:
        return "energy peaks in the middle then recedes"
    return "relatively steady energy throughout"


def describe_track(
    name: str,
    artist: str,
    audio_feats: Optional[Dict[str, Any]],
    duration_ms: Optional[int],
    tags: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> str:
    """
    Produce a maximally detailed natural-language description of a track.

    The output is intended to be consumed by another LLM that will
    refine it into a text-conditioned music-generation prompt.  Every
    available numeric feature is included inline so the downstream model
    has full context.  No attempt is made to be concise or polished —
    information density is the goal.
    """
    sections: list[str] = []

    # ── Identity ────────────────────────────────────────────────
    header = f'"{name}" by {artist}'
    if duration_ms:
        m, s = divmod(int(round(duration_ms / 1000)), 60)
        header += f" (full track duration {m}:{s:02d})"
    sections.append(header + ".")

    if not audio_feats and not tags:
        sections.append(
            "No audio preview was available for analysis; only the track "
            "title and artist name are known.  The downstream prompt should "
            "rely on general knowledge of this artist's style."
        )
        return " ".join(sections)

    # ── Tempo & rhythm ──────────────────────────────────────────
    if audio_feats:
        bpm = audio_feats.get("tempo_bpm") or audio_feats.get("tempo_bpm_librosa") or 0
        if bpm > 0:
            reg = audio_feats.get("beat_regularity", 0)
            reg_desc = (
                "very steady/metronomic" if reg > 0.92
                else "fairly regular" if reg > 0.75
                else "somewhat loose or swung" if reg > 0.5
                else "free-tempo/rubato"
            )
            sections.append(
                f"TEMPO & RHYTHM: {_tempo_descriptor(bpm)} at {bpm:.1f} BPM. "
                f"Beat regularity={reg:.3f} ({reg_desc}). "
            )

        onset_m = audio_feats.get("onset_strength_mean", 0)
        onset_s = audio_feats.get("onset_strength_std", 0)
        percussiveness = (
            "highly percussive, dense transients"
            if onset_m > 2.0
            else "moderately percussive with clear rhythmic attacks"
            if onset_m > 1.0
            else "gentle rhythmic presence, few sharp attacks"
            if onset_m > 0.3
            else "minimal percussive energy; sustained/ambient texture"
        )
        sections.append(
            f"Onset strength mean={onset_m:.3f}, std={onset_s:.3f} "
            f"({percussiveness})."
        )

    # ── Key & tonality ──────────────────────────────────────────
    if audio_feats and audio_feats.get("estimated_key"):
        key = audio_feats["estimated_key"]
        mode = audio_feats.get("mode", "unknown")
        conf = audio_feats.get("mode_confidence", 0)
        chroma = audio_feats.get("chroma_mean", [])
        chroma_str = ""
        if chroma and len(chroma) == 12:
            # Show top-3 pitch classes by energy for harmonic colour
            pairs = sorted(
                zip(_PITCH_CLASSES, chroma), key=lambda x: x[1], reverse=True
            )
            top3 = ", ".join(f"{n}={v:.3f}" for n, v in pairs[:3])
            chroma_str = f" Strongest pitch classes: {top3}."
        sections.append(
            f"KEY & TONALITY: Estimated {key} {mode} "
            f"(Krumhansl correlation={conf:.3f}).{chroma_str}"
        )

    # ── Dynamics ────────────────────────────────────────────────
    if audio_feats and audio_feats.get("rms_mean") is not None:
        rms_m = audio_feats["rms_mean"]
        rms_s = audio_feats.get("rms_std", 0)
        rms_x = audio_feats.get("rms_max", 0)
        dr = audio_feats.get("dynamic_range_db", 0)
        loudness = (
            "very loud/compressed" if rms_m > 0.12
            else "loud and punchy" if rms_m > 0.08
            else "moderate loudness" if rms_m > 0.04
            else "soft and gentle" if rms_m > 0.015
            else "very quiet/delicate"
        )
        dyn_range = (
            "wide dynamic range with large crescendos/decrescendos"
            if dr > 30
            else "moderate dynamic variation"
            if dr > 15
            else "narrow/compressed dynamic range"
        )
        sections.append(
            f"DYNAMICS: RMS mean={rms_m:.4f}, std={rms_s:.4f}, "
            f"max={rms_x:.4f} ({loudness}). "
            f"Dynamic range={dr:.1f} dB ({dyn_range})."
        )

    # ── Energy arc ──────────────────────────────────────────────
    if audio_feats and audio_feats.get("energy_profile_quarters"):
        q = audio_feats["energy_profile_quarters"]
        q_str = ", ".join(f"Q{i+1}={v:.4f}" for i, v in enumerate(q))
        sections.append(
            f"ENERGY ARC (RMS by quarter of clip): [{q_str}] — "
            f"{_energy_arc(q)}."
        )

    # ── Spectral character ──────────────────────────────────────
    if audio_feats and audio_feats.get("spectral_centroid_mean") is not None:
        c_m = audio_feats["spectral_centroid_mean"]
        c_s = audio_feats.get("spectral_centroid_std", 0)
        bw = audio_feats.get("spectral_bandwidth_mean", 0)
        ro = audio_feats.get("spectral_rolloff_mean", 0)
        fl = audio_feats.get("spectral_flatness_mean", 0)
        contrast = audio_feats.get("spectral_contrast_mean", [])

        brightness = (
            "very bright/airy" if c_m > 4000
            else "bright" if c_m > 3000
            else "balanced mid-range" if c_m > 2000
            else "warm/dark" if c_m > 1200
            else "very dark/bass-heavy"
        )
        width = (
            "spectrally wide/rich" if bw > 2500
            else "spectrally narrow/focused" if bw < 1200
            else "moderate spectral width"
        )
        noisiness = (
            "noisy/textured" if fl > 0.1
            else "slightly noisy" if fl > 0.02
            else "clean and tonal" if fl < 0.005
            else "fairly clean"
        )

        spec_desc = (
            f"SPECTRAL CHARACTER: Centroid mean={c_m:.0f} Hz (std={c_s:.0f} Hz) "
            f"— {brightness}. Bandwidth={bw:.0f} Hz ({width}). "
            f"Rolloff(85%)={ro:.0f} Hz. Flatness={fl:.6f} ({noisiness})."
        )
        if contrast and len(contrast) == 7:
            bands = ["sub-bass", "bass", "low-mid", "mid", "upper-mid", "presence", "brilliance"]
            c_str = ", ".join(f"{bands[i]}={v:.1f}" for i, v in enumerate(contrast))
            spec_desc += f" Spectral contrast by band: [{c_str}]."
        sections.append(spec_desc)

    # ── Timbre (MFCCs) ──────────────────────────────────────────
    if audio_feats and audio_feats.get("mfcc_mean"):
        mm = audio_feats["mfcc_mean"]
        ms = audio_feats.get("mfcc_std", [])
        # MFCC 0 ≈ overall energy; 1-4 encode broad timbral shape
        sections.append(
            f"TIMBRE (MFCCs 0-12 mean): [{', '.join(f'{v:.1f}' for v in mm)}]. "
            f"MFCC std: [{', '.join(f'{v:.1f}' for v in ms)}]. "
            f"MFCC-0 (energy)={mm[0]:.1f}; MFCC-1 (spectral slope)={mm[1]:.1f}; "
            f"MFCC-2 (spectral curvature)={mm[2]:.1f}."
        )

    # ── Texture: harmonic ratio & ZCR ───────────────────────────
    if audio_feats:
        hr = audio_feats.get("harmonic_ratio")
        zcr = audio_feats.get("zero_crossing_rate_mean")
        parts = []
        if hr is not None:
            hp_desc = (
                "predominantly harmonic/melodic" if hr > 0.85
                else "balanced harmonic and percussive" if hr > 0.6
                else "percussion-dominated"
            )
            parts.append(
                f"Harmonic-to-total energy ratio={hr:.3f} ({hp_desc})"
            )
        if zcr is not None:
            zcr_desc = (
                "noisy/distorted/breathy"
                if zcr > 0.15
                else "moderate texture"
                if zcr > 0.05
                else "smooth/sustained tones"
            )
            parts.append(f"Zero-crossing rate={zcr:.5f} ({zcr_desc})")
        if parts:
            sections.append("TEXTURE: " + ". ".join(parts) + ".")

    # ── Tags: genre, instrument, vocal ──────────────────────────
    if tags:
        genres = tags.get("genre", [])
        if genres:
            names = [f'{t["label"]}={t["prob"]:.1%}' for t in genres[:5]]
            sections.append("DETECTED GENRES (PANNs AudioSet, confidence): " + ", ".join(names) + ".")

        instruments = tags.get("instrument", [])
        if instruments:
            names = [f'{t["label"]}={t["prob"]:.1%}' for t in instruments[:5]]
            sections.append("DETECTED INSTRUMENTS: " + ", ".join(names) + ".")
        else:
            sections.append(
                "INSTRUMENTS: No specific instruments detected above threshold "
                "(may be heavily mixed or synthetic/electronic)."
            )

        vocals = tags.get("vocal", [])
        vocal_max = tags.get("_vocal_prob_max", 0.0)

        # Check if detected genres imply vocals are present
        genre_names = {g["label"] for g in genres}
        genre_implies_vocals = bool(genre_names & _VOCAL_GENRES)

        if vocals:
            names = [f'{t["label"]}={t["prob"]:.1%}' for t in vocals[:3]]
            sections.append(
                f"VOCALS: Detected — " + ", ".join(names)
                + f". Peak vocal probability across windows={vocal_max:.3f}."
            )
        elif genre_implies_vocals:
            sections.append(
                f"VOCALS: Not strongly detected in the 30-second preview "
                f"(peak vocal prob={vocal_max:.3f}), but detected genres "
                f"({', '.join(genre_names & _VOCAL_GENRES)}) strongly imply "
                f"vocals are present in the full track.  Assume vocals present."
            )
        elif vocal_max > 0.005:
            sections.append(
                f"VOCALS: Faint vocal signal (peak={vocal_max:.3f}); "
                f"may contain vocals in the full track but the 30-second "
                f"preview is ambiguous."
            )
        else:
            sections.append(
                f"VOCALS: Likely instrumental — no vocal signal detected "
                f"(peak vocal prob={vocal_max:.3f})."
            )

    # ── Caveat about preview limitations ────────────────────────
    sections.append(
        "NOTE: All features are extracted from a 30-second Deezer preview "
        "clip, not the full track.  The preview may not be representative "
        "of the entire song (e.g. intros, outros, key changes, tempo "
        "shifts in the full track are not captured)."
    )

    return " ".join(sections)


# Backward-compatible alias
def one_liner_from_features(
    name: str,
    artist: str,
    audio_feats: Optional[Dict[str, Any]],
    duration_ms: Optional[int],
    tags: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> str:
    return describe_track(name, artist, audio_feats, duration_ms, tags)

def process_ytm_playlist(playlist_id: str, max_tracks: int = 50) -> Dict[str, Any]:
    tracks = fetch_ytm_playlist_tracks(playlist_id, limit=max_tracks)

    out_tracks = []
    preview_ok = 0

    for i, t in enumerate(tracks, start=1):
        name = t["name"]
        artist = t["artist"]
        duration_ms = t.get("duration_ms")

        record: Dict[str, Any] = {
            "ytm": t,
            "source": "ytm_metadata_only",
            "deezer_match": None,
            "preview_audio": None,
            "audio_features": None,
            "tags": None,
            "description": None,
        }

        try:
            # be polite with Deezer
            if i > 1:
                time.sleep(0.25)

            dz = try_deezer_preview(name, artist, duration_ms)
            record["deezer_match"] = dz.get("deezer")

            if dz.get("found"):
                record["source"] = "ytm_metadata + deezer_preview"
                record["preview_audio"] = {
                    "preview_url": dz.get("preview_url"),
                    "audio_path": dz.get("audio_path"),
                }
                record["audio_features"] = extract_audio_features(dz["audio_path"])
                record["tags"] = extract_tags(dz["audio_path"])
                preview_ok += 1
        except Exception as e:
            record["error"] = str(e)

        record["description"] = describe_track(
            name=name,
            artist=artist,
            audio_feats=record["audio_features"],
            duration_ms=duration_ms,
            tags=record["tags"],
        )

        out_tracks.append(record)

    summary = {
        "playlist_id": playlist_id,
        "tracks_processed": len(out_tracks),
        "deezer_preview_found": preview_ok,
        "metadata_only": len(out_tracks) - preview_ok,
    }

    return {"summary": summary, "tracks": out_tracks}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("playlist_id", help="YouTube Music playlistId (not a URL)")
    p.add_argument("--max_tracks", type=int, default=50)
    args = p.parse_args()

    data = process_ytm_playlist(args.playlist_id, max_tracks=args.max_tracks)
    out_path = os.path.join(OUT_DIR, f"{args.playlist_id}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Wrote:", out_path)
    print(json.dumps(data["summary"], indent=2))
