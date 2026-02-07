"""
modal_app.py – Serverless GPU audio analysis + GPT-4 playlist summarisation.
=============================================================================
Deploys on Modal with a T4 GPU for PANNs inference and calls OpenAI GPT-4
to produce a unified music-generation prompt from per-track descriptions.

Deploy:   modal deploy models/modal_app.py
Serve:    modal serve models/modal_app.py   (hot-reload dev mode)
"""

import modal

# ── Modal image with all Python deps ────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "wget")
    .pip_install(
        "librosa==0.10.2",
        "numpy<2",
        "torch==2.2.2",
        "requests",
        "openai",
        "panns-inference",
        "soundfile",
        "fastapi[standard]",
    )
    # panns_inference reads the CSV at *import time*, so it must exist in the image
    # Also pre-download the ~300MB model checkpoint to avoid runtime download on cold start
    .run_commands(
        "mkdir -p /root/panns_data",
        "wget -q -O /root/panns_data/class_labels_indices.csv "
        "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv",
        'wget -q -O "/root/panns_data/Cnn14_mAP=0.431.pth" '
        '"https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"',
    )
)

app = modal.App("musicgen-analyzer")

# ── Reusable analysis functions (inlined from ytm_to_deezer.py) ────

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MUSIC_LABELS = {
    "genre": [
        216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229,
        230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
        244, 245, 246, 247, 248, 249, 250, 251, 253, 254, 255, 257, 259, 263,
        264, 265, 274,
    ],
    "instrument": [
        140, 141, 142, 143, 153, 154, 156, 158, 161, 162, 163, 164, 165, 168,
        185, 189, 191, 194, 195, 196,
    ],
    "vocal": [27, 32, 33, 34, 35],
}

# Per-category thresholds for better accuracy
_CATEGORY_THRESHOLDS = {
    "genre": 0.02,
    "instrument": 0.03,
    "vocal": 0.01,  # Lower threshold to catch vocals in intros/outros
}

# Genres that typically contain vocals
_VOCAL_GENRES = {
    "Pop music", "Hip hop music", "Rock music", "Heavy metal",
    "Punk rock", "Rock and roll", "Rhythm and blues", "Soul music",
    "Reggae", "Country", "Folk music", "Blues", "Vocal music",
    "Funk", "Disco", "Gospel music", "Grunge", "Ska",
    "Beatboxing", "A capella",
}

# Lazy-loaded singleton tagger
_panns_tagger = None
_panns_labels = []


def _get_tagger():
    """Return the PANNs AudioTagging model (loaded once, on GPU if available)."""
    global _panns_tagger, _panns_labels
    if _panns_tagger is None:
        import csv, os, requests
        from panns_inference import AudioTagging
        import torch

        # Prepare directory and download labels CSV
        panns_dir = os.path.join(os.path.expanduser("~"), "panns_data")
        os.makedirs(panns_dir, exist_ok=True)
        labels_csv = os.path.join(panns_dir, "class_labels_indices.csv")
        
        if not os.path.exists(labels_csv):
            csv_url = "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
            try:
                print(f"[modal] Downloading PANNs labels CSV...")
                response = requests.get(csv_url, timeout=30)
                response.raise_for_status()
                with open(labels_csv, 'wb') as f:
                    f.write(response.content)
                print(f"[modal] PANNs labels downloaded successfully")
            except Exception as e:
                print(f"[modal] ERROR: Could not download PANNs labels: {e}")
                # Return empty to allow processing to continue without tags
                return None, []
        
        # Load labels
        try:
            with open(labels_csv) as f:
                reader = csv.reader(f)
                next(reader)
                _panns_labels = [row[2] for row in reader]
            print(f"[modal] Loaded {len(_panns_labels)} PANNs labels")
        except Exception as e:
            print(f"[modal] ERROR: Could not read PANNs labels: {e}")
            return None, []

        # Initialize tagger (this will download checkpoint if needed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            print(f"[modal] Initializing PANNs tagger on {device}...")
            _panns_tagger = AudioTagging(checkpoint_path=None, device=device)
            print(f"[modal] PANNs tagger initialized successfully")
        except Exception as e:
            print(f"[modal] ERROR: Could not initialize PANNs tagger: {e}")
            return None, []

    return _panns_tagger, _panns_labels


def download_file(url, out_path, timeout=30):
    import os, requests as req
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return out_path
    r = req.get(url, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


def extract_audio_features(audio_path, sr=22050):
    import numpy as np, librosa

    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    duration_sec = float(len(y) / sr)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo).flat[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_regularity = (
        float(1.0 - np.std(np.diff(beat_times)) / (np.mean(np.diff(beat_times)) + 1e-8))
        if len(beat_times) > 1 else 0.0
    )
    onset = librosa.onset.onset_strength(y=y, sr=sr)

    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma_cq, axis=1)
    key_idx = int(np.argmax(chroma_mean))
    key_name = _PITCH_CLASSES[key_idx]
    _major = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    _minor = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    corr_major = float(np.corrcoef(np.roll(chroma_mean, -key_idx), _major)[0, 1])
    corr_minor = float(np.corrcoef(np.roll(chroma_mean, -key_idx), _minor)[0, 1])
    mode = "major" if corr_major >= corr_minor else "minor"
    mode_confidence = round(float(max(corr_major, corr_minor)), 3)

    rms = librosa.feature.rms(y=y)[0]
    rms_floor = np.maximum(rms, 1e-10)
    dynamic_range_db = float(20.0 * np.log10(np.max(rms_floor) / np.min(rms_floor)))
    quarter = len(rms) // 4 or 1
    energy_profile = [float(np.mean(rms[i * quarter:(i + 1) * quarter])) for i in range(4)]

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    y_harm, _ = librosa.effects.hpss(y)
    harmonic_ratio = float(np.sum(y_harm ** 2) / (np.sum(y ** 2) + 1e-10))

    return {
        "tempo_bpm": tempo,
        "beat_regularity": round(beat_regularity, 4),
        "onset_strength_mean": round(float(np.mean(onset)), 4),
        "onset_strength_std": round(float(np.std(onset)), 4),
        "duration_sec": round(duration_sec, 2),
        "estimated_key": key_name,
        "mode": mode,
        "mode_confidence": mode_confidence,
        "chroma_mean": [round(float(c), 4) for c in chroma_mean],
        "rms_mean": round(float(np.mean(rms)), 5),
        "rms_std": round(float(np.std(rms)), 5),
        "rms_max": round(float(np.max(rms)), 5),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "energy_profile_quarters": [round(e, 5) for e in energy_profile],
        "spectral_centroid_mean": round(float(np.mean(centroid)), 2),
        "spectral_centroid_std": round(float(np.std(centroid)), 2),
        "spectral_bandwidth_mean": round(float(np.mean(bandwidth)), 2),
        "spectral_rolloff_mean": round(float(np.mean(rolloff)), 2),
        "spectral_flatness_mean": round(float(np.mean(flatness)), 6),
        "spectral_contrast_mean": [round(float(c), 4) for c in np.mean(contrast, axis=1)],
        "mfcc_mean": [round(float(c), 4) for c in np.mean(mfcc, axis=1)],
        "mfcc_std": [round(float(c), 4) for c in np.std(mfcc, axis=1)],
        "zero_crossing_rate_mean": round(float(np.mean(zcr)), 6),
        "harmonic_ratio": round(harmonic_ratio, 4),
    }


def extract_tags(audio_path, top_n=5, vocal_window_sec=10.0, vocal_hop_sec=5.0):
    """Extract tags with windowed vocal scanning for better vocal detection."""
    import numpy as np, librosa, torch
    
    tagger, labels = _get_tagger()
    if tagger is None or not labels:
        print(f"[modal] Skipping tagging - tagger not available")
        result = {cat: [] for cat in MUSIC_LABELS.keys()}
        result["_vocal_prob_max"] = 0.0
        return result

    SR_PANNS = 32000
    y, _ = librosa.load(audio_path, sr=SR_PANNS, mono=True)

    # Full-clip inference for genre & instrument
    waveform = y[np.newaxis, :]
    with torch.no_grad():
        clipwise_output, _ = tagger.inference(waveform)
    full_probs = clipwise_output[0]

    # Windowed vocal scan - better at catching vocals in intros/outros
    win_samples = int(vocal_window_sec * SR_PANNS)
    hop_samples = int(vocal_hop_sec * SR_PANNS)
    starts = list(range(0, max(1, len(y) - win_samples + 1), hop_samples))
    if not starts:
        starts = [0]

    vocal_indices = MUSIC_LABELS["vocal"]
    best_vocal = {idx: 0.0 for idx in vocal_indices}

    for s in starts:
        chunk = y[s : s + win_samples]
        if len(chunk) < SR_PANNS:  # skip tiny tail (<1s)
            continue
        wav = chunk[np.newaxis, :]
        with torch.no_grad():
            co, _ = tagger.inference(wav)
        p = co[0]
        for idx in vocal_indices:
            best_vocal[idx] = max(best_vocal[idx], float(p[idx]))

    # Assemble results with category-specific thresholds
    result = {}
    for category, indices in MUSIC_LABELS.items():
        thresh = _CATEGORY_THRESHOLDS.get(category, 0.03)
        cat_tags = []
        for idx in indices:
            if category == "vocal":
                p = best_vocal.get(idx, 0.0)  # use windowed max
            else:
                p = float(full_probs[idx])  # use full-clip
            if p >= thresh:
                cat_tags.append({"label": labels[idx], "prob": round(p, 4)})
        cat_tags.sort(key=lambda x: x["prob"], reverse=True)
        result[category] = cat_tags[:top_n]

    # Expose peak vocal prob for downstream reasoning
    result["_vocal_prob_max"] = round(max(best_vocal.values()) if best_vocal else 0.0, 4)
    return result


# ── Description helpers ─────────────────────────────────────────────

def _tempo_descriptor(bpm):
    if bpm < 40: return "very slow (grave)"
    if bpm < 66: return "slow (largo/adagio)"
    if bpm < 76: return "walking pace (andante)"
    if bpm < 108: return "moderate (moderato)"
    if bpm < 120: return "lively (allegretto)"
    if bpm < 156: return "fast (allegro)"
    if bpm < 176: return "very fast (vivace)"
    return "extremely fast (presto)"

def _dynamics_descriptor(rms_mean, dynamic_range_db):
    parts = []
    if rms_mean > 0.12: parts.append("very loud and compressed")
    elif rms_mean > 0.08: parts.append("loud and punchy")
    elif rms_mean > 0.04: parts.append("moderate loudness")
    elif rms_mean > 0.015: parts.append("soft and gentle")
    else: parts.append("very quiet and delicate")
    if dynamic_range_db > 30: parts.append("with a wide dynamic range")
    elif dynamic_range_db > 15: parts.append("with moderate dynamic variation")
    else: parts.append("with a narrow dynamic range")
    return ", ".join(parts)

def _spectral_descriptor(centroid, bandwidth, rolloff, flatness):
    parts = []
    if centroid > 4000: parts.append("very bright, airy timbre")
    elif centroid > 3000: parts.append("bright timbre")
    elif centroid > 2000: parts.append("balanced, mid-range timbre")
    elif centroid > 1200: parts.append("warm, slightly dark timbre")
    else: parts.append("dark, bass-heavy timbre")
    if bandwidth > 2500: parts.append("spectrally wide and rich")
    elif bandwidth < 1200: parts.append("spectrally narrow and focused")
    if flatness > 0.1: parts.append("noisy/textured sound")
    elif flatness < 0.005: parts.append("clean, tonal sound")
    return ", ".join(parts)

def _energy_arc(quarters):
    if not quarters or len(quarters) < 4: return "steady energy"
    q = quarters
    if q[3] > q[0] * 1.3: return "energy builds throughout"
    if q[3] < q[0] * 0.7: return "energy fades over time"
    if max(q[1], q[2]) > max(q[0], q[3]) * 1.3: return "energy peaks in the middle"
    return "relatively steady energy throughout"


def describe_track(name, artist, audio_feats, duration_ms, tags=None):
    sections = []
    header = f'"{name}" by {artist}'
    if duration_ms:
        m, s = divmod(int(round(duration_ms / 1000)), 60)
        header += f" ({m}:{s:02d})"
    sections.append(header + ".")

    if not audio_feats and not tags:
        sections.append("No audio preview was available; only metadata is known.")
        return " ".join(sections)

    if audio_feats:
        bpm = audio_feats.get("tempo_bpm", 0)
        if bpm > 0:
            reg = audio_feats.get("beat_regularity", 0)
            reg_desc = (
                "very steady, metronomic beat" if reg > 0.92
                else "fairly regular beat" if reg > 0.75
                else "somewhat loose or swung rhythm" if reg > 0.5
                else "free-tempo or rubato feel"
            )
            sections.append(f"Tempo is {_tempo_descriptor(bpm)} at ~{int(round(bpm))} BPM with a {reg_desc}.")
        onset_m = audio_feats.get("onset_strength_mean", 0)
        if onset_m > 2.0: sections.append("Highly percussive with frequent transients.")
        elif onset_m > 1.0: sections.append("Moderately percussive with clear rhythmic attacks.")
        elif onset_m > 0.3: sections.append("Gentle rhythmic presence.")
        else: sections.append("Minimal percussive elements; sustained or ambient texture.")

    if audio_feats and audio_feats.get("estimated_key"):
        key = audio_feats["estimated_key"]
        mode = audio_feats.get("mode", "unknown")
        conf = audio_feats.get("mode_confidence", 0)
        conf_word = "strongly" if conf > 0.85 else "likely" if conf > 0.65 else "possibly"
        sections.append(f"Key: {conf_word} {key} {mode}.")

    if audio_feats and audio_feats.get("rms_mean") is not None:
        sections.append("Dynamics: " + _dynamics_descriptor(audio_feats["rms_mean"], audio_feats.get("dynamic_range_db", 0)) + ".")
    if audio_feats and audio_feats.get("energy_profile_quarters"):
        sections.append("Energy arc: " + _energy_arc(audio_feats["energy_profile_quarters"]) + ".")
    if audio_feats and audio_feats.get("spectral_centroid_mean") is not None:
        sections.append("Spectral character: " + _spectral_descriptor(
            audio_feats["spectral_centroid_mean"],
            audio_feats.get("spectral_bandwidth_mean", 1500),
            audio_feats.get("spectral_rolloff_mean", 3000),
            audio_feats.get("spectral_flatness_mean", 0.01),
        ) + ".")
    if audio_feats and audio_feats.get("harmonic_ratio") is not None:
        hr = audio_feats["harmonic_ratio"]
        if hr > 0.85: sections.append("Predominantly harmonic/melodic content.")
        elif hr > 0.6: sections.append("Good balance of melodic and percussive elements.")
        else: sections.append("Percussion-dominated mix with strong transient energy.")

    if tags:
        genres = tags.get("genre", [])
        if genres:
            sections.append("Detected genres: " + ", ".join(f'{t["label"]} ({t["prob"]:.0%})' for t in genres[:5]) + ".")

        # Sub-genre inference: map broad PANNs genres to more specific styles
        # based on audio features (PANNs has no trap, lo-fi, drill, synthwave, etc.)
        genre_labels = {g["label"] for g in genres}
        inferred_subgenres = []
        bpm = audio_feats.get("tempo_bpm", 0) if audio_feats else 0
        centroid = audio_feats.get("spectral_centroid_mean", 2000) if audio_feats else 2000
        flatness = audio_feats.get("spectral_flatness_mean", 0.01) if audio_feats else 0.01
        hr = audio_feats.get("harmonic_ratio", 0.5) if audio_feats else 0.5
        has_808 = any(t["label"] in ("Drum machine", "Bass drum") for t in tags.get("instrument", []))
        has_synth = any(t["label"] in ("Synthesizer", "Electronic organ") for t in tags.get("instrument", []))

        if "Hip hop music" in genre_labels:
            if has_808 and bpm >= 130 and bpm <= 170:
                inferred_subgenres.append("trap")
            elif has_808 and bpm >= 130 and centroid > 3000:
                inferred_subgenres.append("drill")
            elif bpm < 100 and flatness < 0.02:
                inferred_subgenres.append("lo-fi hip hop")
            elif bpm >= 85 and bpm <= 115:
                inferred_subgenres.append("boom bap")
        if "Electronic music" in genre_labels or "Electronic dance music" in genre_labels:
            if bpm >= 120 and bpm <= 135 and has_synth:
                inferred_subgenres.append("house")
            elif bpm >= 135 and bpm <= 150:
                inferred_subgenres.append("techno")
            elif bpm >= 80 and bpm <= 115 and has_synth and centroid > 3000:
                inferred_subgenres.append("synthwave")
            elif centroid < 2000 and flatness < 0.01:
                inferred_subgenres.append("ambient electronic")
        if "Rhythm and blues" in genre_labels:
            if has_808 or has_synth:
                inferred_subgenres.append("contemporary R&B")
            else:
                inferred_subgenres.append("classic R&B")
        if "Rock music" in genre_labels and centroid > 3500 and bpm >= 100:
            inferred_subgenres.append("alternative rock")

        if inferred_subgenres:
            sections.append("Inferred sub-genre(s): " + ", ".join(inferred_subgenres) + ".")

        instruments = tags.get("instrument", [])
        if instruments:
            sections.append("Detected instruments: " + ", ".join(f'{t["label"]} ({t["prob"]:.0%})' for t in instruments[:5]) + ".")
        vocals = tags.get("vocal", [])
        vocal_prob_max = tags.get("_vocal_prob_max", 0.0)
        genres = tags.get("genre", [])
        genre_labels = {g["label"] for g in genres}
        has_vocal_genre = bool(genre_labels & _VOCAL_GENRES)
        
        if vocals:
            sections.append("Vocal character: " + ", ".join(f'{t["label"]} ({t["prob"]:.0%})' for t in vocals[:3]) + ".")
        elif has_vocal_genre and vocal_prob_max > 0.005:
            sections.append(f"Possible vocals detected (confidence {vocal_prob_max:.1%}), common in this genre.")
        else:
            sections.append("Likely instrumental (no strong vocal signal detected).")

    return " ".join(sections)


# ── GPT-4 playlist summariser ──────────────────────────────────────

def summarise_playlist_with_gpt4(track_descriptions):
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(track_descriptions))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a music analysis expert. The user will give you per-track audio "
                    "analysis descriptions for an entire playlist. Your job is to synthesise "
                    "them into a SINGLE concise music-generation style prompt that focuses "
                    "exclusively on INSTRUMENTAL and PRODUCTION characteristics.\n\n"
                    "The prompt should capture:\n"
                    "- Dominant genre(s) and sub-genres\n"
                    "- Overall mood and emotional arc\n"
                    "- Typical tempo range and rhythmic feel\n"
                    "- Key instrumentation (guitars, synths, drums, bass, strings, etc.)\n"
                    "- Production style and sonic texture (lo-fi, polished, reverb-heavy, etc.)\n"
                    "- Energy level, dynamics, and arrangement style\n\n"
                    "Do NOT mention vocals, singing, or vocal characteristics — this prompt "
                    "will be used as an instrumental style reference for music generation.\n\n"
                    "Output ONLY the instrumental style prompt, nothing else. "
                    "Keep it under 200 words. Be specific and vivid."
                ),
            },
            {
                "role": "user",
                "content": f"Here are the per-track descriptions for a playlist:\n\n{numbered}",
            },
        ],
        temperature=0.7,
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()


# ── Modal endpoint ──────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=600,
)
@modal.fastapi_endpoint(method="POST")
def analyze_playlist(data: dict):
    import os, hashlib, tempfile

    tracks = data.get("tracks", [])
    if not tracks:
        return {"error": "tracks list is required"}

    tmp_dir = tempfile.mkdtemp(prefix="musicgen_")
    results = []
    descriptions = []

    for t in tracks:
        name = t.get("name", "")
        artist = t.get("artist", "")
        duration_ms = t.get("duration_ms")
        preview_url = t.get("preview_url")

        record = {
            "name": name,
            "artist": artist,
            "duration_ms": duration_ms,
            "audio_features": None,
            "tags": None,
            "description": None,
        }

        audio_path = None
        if preview_url:
            h = hashlib.sha1(preview_url.encode("utf-8")).hexdigest()[:12]
            audio_path = os.path.join(tmp_dir, f"preview_{h}.mp3")
            try:
                download_file(preview_url, audio_path)
            except Exception as e:
                print(f"[modal] download failed for {name}: {e}")
                audio_path = None

        if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1024:
            try:
                record["audio_features"] = extract_audio_features(audio_path)
            except Exception as e:
                print(f"[modal] feature extraction failed for {name}: {e}")
            try:
                record["tags"] = extract_tags(audio_path)
            except Exception as e:
                print(f"[modal] tagging failed for {name}: {e}")
                import traceback
                traceback.print_exc()
                # Set empty tags so processing continues
                record["tags"] = {cat: [] for cat in MUSIC_LABELS.keys()}

        desc = describe_track(
            name=name,
            artist=artist,
            audio_feats=record["audio_features"],
            duration_ms=duration_ms,
            tags=record["tags"],
        )
        record["description"] = desc
        descriptions.append(desc)
        results.append(record)

    # Generate unified playlist summary via GPT-4
    playlist_summary = None
    analyzed_descriptions = [d for d in descriptions if "No audio preview" not in d]
    if analyzed_descriptions:
        try:
            playlist_summary = summarise_playlist_with_gpt4(analyzed_descriptions)
        except Exception as e:
            print(f"[modal] GPT-4 summary failed: {e}")
            playlist_summary = f"Summary generation failed: {e}"

    return {"results": results, "playlist_summary": playlist_summary}
