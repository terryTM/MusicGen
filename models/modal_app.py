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
    .apt_install("ffmpeg", "libsndfile1")
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
)

app = modal.App("musicgen-analyzer")

# ── Reusable analysis functions (inlined from ytm_to_deezer.py) ────

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MUSIC_LABELS = {
    "genre": [
        216, 217, 219, 220, 221, 223, 224, 225, 226, 227,
        228, 229, 233, 235, 237, 239, 243, 244, 245, 251, 254,
    ],
    "instrument": [
        140, 141, 142, 143, 153, 154, 156, 162, 163, 164, 165, 168, 191, 194, 196,
    ],
    "vocal": [27, 32, 33, 34, 35],
}


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


def extract_tags(audio_path, top_n=5, threshold=0.03):
    import csv, os, numpy as np, librosa, torch
    from panns_inference import AudioTagging

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tagger = AudioTagging(checkpoint_path=None, device=device)

    labels_csv = os.path.join(os.path.expanduser("~"), "panns_data", "class_labels_indices.csv")
    with open(labels_csv) as f:
        reader = csv.reader(f)
        next(reader)
        labels = [row[2] for row in reader]

    y, _ = librosa.load(audio_path, sr=32000, mono=True)
    waveform = y[np.newaxis, :]

    with torch.no_grad():
        clipwise_output, _ = tagger.inference(waveform)
    probs = clipwise_output[0]

    result = {}
    for category, indices in MUSIC_LABELS.items():
        cat_tags = []
        for idx in indices:
            p = float(probs[idx])
            if p >= threshold:
                cat_tags.append({"label": labels[idx], "prob": round(p, 4)})
        cat_tags.sort(key=lambda x: x["prob"], reverse=True)
        result[category] = cat_tags[:top_n]
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
        instruments = tags.get("instrument", [])
        if instruments:
            sections.append("Detected instruments: " + ", ".join(f'{t["label"]} ({t["prob"]:.0%})' for t in instruments[:5]) + ".")
        vocals = tags.get("vocal", [])
        if vocals:
            sections.append("Vocal character: " + ", ".join(f'{t["label"]} ({t["prob"]:.0%})' for t in vocals[:3]) + ".")
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
                    "them into a SINGLE concise music-generation prompt (the kind you would "
                    "feed to MusicGen, Suno, or Udio).\n\n"
                    "The prompt should capture:\n"
                    "- Dominant genre(s) and sub-genres\n"
                    "- Overall mood and emotional arc\n"
                    "- Typical tempo range and rhythmic feel\n"
                    "- Key instrumentation and production style\n"
                    "- Vocal characteristics\n"
                    "- Energy level and dynamics\n\n"
                    "Output ONLY the music-generation prompt, nothing else. "
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
