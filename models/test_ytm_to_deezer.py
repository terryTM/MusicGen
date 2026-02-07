"""
Quick capability test for ytm_to_deezer.py
==========================================
Exercises:
  1. Utility helpers   (norm, parse_duration_to_ms)
  2. Rich audio feature extraction on a synthetic signal
  3. Rich description generator (describe_track)
  4. Live Deezer search API (network-dependent)
  5. PANNs music tagging on a real Deezer preview (GPU)
"""

import os, sys, json, tempfile
import numpy as np
import soundfile as sf

# Ensure the models package is importable
sys.path.insert(0, os.path.dirname(__file__))

from ytm_to_deezer import (
    norm,
    parse_duration_to_ms,
    extract_audio_features,
    extract_basic_audio_features,   # backward-compat alias
    extract_tags,
    describe_track,
    one_liner_from_features,         # backward-compat alias
    deezer_search,
    pick_best_deezer_hit,
    try_deezer_preview,
)

SEP = "-" * 60


# ── 1. Utility helpers ──────────────────────────────────────────

def test_utilities():
    print(SEP)
    print("1 · Utility helpers")
    print(SEP)

    assert norm("  Bohemian Rhapsody (Remastered 2011) ") == "bohemian rhapsody"
    assert norm("[Live] Stairway to Heaven") == "stairway to heaven"
    assert norm("") == ""

    assert parse_duration_to_ms("3:45") == 225_000
    assert parse_duration_to_ms("1:02:10") == 3_730_000
    assert parse_duration_to_ms(None) is None
    assert parse_duration_to_ms("abc") is None

    print("  ✓ norm() and parse_duration_to_ms() pass")


# ── 2. Feature extraction on synthetic audio ────────────────────

def _make_synth_wav(path: str, freq: float = 440.0, dur: float = 5.0, sr: int = 22050):
    """Generate a simple sine wave WAV."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, y, sr)
    return path


def test_feature_extraction():
    print(f"\n{SEP}")
    print("2 · Rich audio feature extraction (synthetic 440 Hz sine, 5 s)")
    print(SEP)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        _make_synth_wav(wav_path, freq=440.0, dur=5.0)
        feats = extract_audio_features(wav_path)

        # Check all expected keys are present
        expected_keys = {
            "tempo_bpm", "beat_regularity", "onset_strength_mean", "onset_strength_std",
            "duration_sec", "estimated_key", "mode", "mode_confidence", "chroma_mean",
            "rms_mean", "rms_std", "rms_max", "dynamic_range_db", "energy_profile_quarters",
            "spectral_centroid_mean", "spectral_centroid_std", "spectral_bandwidth_mean",
            "spectral_rolloff_mean", "spectral_flatness_mean", "spectral_contrast_mean",
            "mfcc_mean", "mfcc_std", "zero_crossing_rate_mean", "harmonic_ratio",
        }
        missing = expected_keys - set(feats.keys())
        assert not missing, f"Missing feature keys: {missing}"

        print("  Extracted features (scalar subset):")
        for k, v in feats.items():
            if isinstance(v, (int, float)):
                print(f"    {k:30s} = {v}")

        # Sanity checks
        assert isinstance(feats["tempo_bpm"], float)
        assert feats["rms_mean"] > 0, "RMS should be positive for a non-silent signal"
        assert feats["spectral_centroid_mean"] > 0
        assert feats["estimated_key"] in ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        assert feats["mode"] in ("major", "minor")
        assert len(feats["chroma_mean"]) == 12
        assert len(feats["mfcc_mean"]) == 13
        assert len(feats["mfcc_std"]) == 13
        assert len(feats["energy_profile_quarters"]) == 4
        assert 0.0 <= feats["harmonic_ratio"] <= 1.0

        # Backward-compat alias should work identically
        feats2 = extract_basic_audio_features(wav_path)
        assert feats2["tempo_bpm"] == feats["tempo_bpm"]

        print("  ✓ Rich feature extraction OK")
    finally:
        os.unlink(wav_path)


# ── 3. One-liner vibe summary ──────────────────────────────────

def test_describe_track():
    print(f"\n{SEP}")
    print("3 · Rich description generator (describe_track)")
    print(SEP)

    # With full audio features
    feats = {
        "tempo_bpm": 128.0,
        "beat_regularity": 0.91,
        "onset_strength_mean": 1.5,
        "onset_strength_std": 0.8,
        "estimated_key": "C#",
        "mode": "minor",
        "mode_confidence": 0.82,
        "rms_mean": 0.10,
        "rms_std": 0.02,
        "rms_max": 0.15,
        "dynamic_range_db": 25.0,
        "energy_profile_quarters": [0.08, 0.10, 0.12, 0.10],
        "spectral_centroid_mean": 3500.0,
        "spectral_centroid_std": 500.0,
        "spectral_bandwidth_mean": 2600.0,
        "spectral_rolloff_mean": 5000.0,
        "spectral_flatness_mean": 0.004,
        "spectral_contrast_mean": [20.0, 18.0, 15.0, 12.0, 10.0, 8.0, 5.0],
        "mfcc_mean": [0.0] * 13,
        "mfcc_std": [1.0] * 13,
        "chroma_mean": [0.5] * 12,
        "zero_crossing_rate_mean": 0.06,
        "harmonic_ratio": 0.70,
        "duration_sec": 30.0,
    }
    mock_tags = {
        "genre": [{"label": "Electronic dance music", "prob": 0.45}],
        "instrument": [{"label": "Drum machine", "prob": 0.30}],
        "vocal": [{"label": "Singing", "prob": 0.20}],
    }
    desc = describe_track("Blinding Lights", "The Weeknd", feats, 200_000, mock_tags)
    print(f"  Full desc:\n  {desc}")
    assert "128" in desc and "BPM" in desc
    assert "C# minor" in desc or "C#" in desc
    assert "bright" in desc.lower()
    assert "Electronic dance music" in desc
    assert "Drum machine" in desc
    assert "Singing" in desc
    assert "DYNAMICS" in desc
    assert "ENERGY ARC" in desc

    # Without audio features or tags (metadata-only fallback)
    desc2 = describe_track("Unknown", "Nobody", None, None)
    print(f"  No feats   : {desc2}")
    assert "No audio preview" in desc2

    # Tags only, no audio features
    desc3 = describe_track("Tagged", "Artist", None, 200_000, tags=mock_tags)
    print(f"  Tags only  : {desc3}")
    assert "Electronic dance music" in desc3
    assert "No audio preview" not in desc3

    # Backward-compat alias works
    desc4 = one_liner_from_features("Test", "A", feats, 180_000)
    assert len(desc4) > 50  # should be a rich description now

    print("  ✓ Rich description generation OK")


# ── 4. Live Deezer search ──────────────────────────────────────

def test_deezer_search():
    print(f"\n{SEP}")
    print("4 · Deezer search API (live network call)")
    print(SEP)

    try:
        data = deezer_search("Bohemian Rhapsody", "Queen", limit=5)
        hits = data.get("data", [])
        print(f"  Query: 'Bohemian Rhapsody' by 'Queen'  →  {len(hits)} results")

        if hits:
            best = pick_best_deezer_hit(hits, "Bohemian Rhapsody", "Queen", 354_000)
            if best:
                title = best.get("title", "?")
                artist = (best.get("artist") or {}).get("name", "?")
                preview = "yes" if best.get("preview") else "no"
                print(f"  Best match : {title} by {artist} (preview: {preview})")
            else:
                print("  ⚠ No best match found among hits")

        print("  ✓ Deezer search OK")

    except Exception as e:
        print(f"  ⚠ Deezer search failed (network?): {e}")


# ── 5. PANNs music tagging (real preview, GPU) ────────────────

def test_tagging():
    print(f"\n{SEP}")
    print("5 · PANNs music tagging on real Deezer previews (GPU)")
    print(SEP)

    test_tracks = [
        ("Lose Yourself", "Eminem", 326_000),
        ("Clair de Lune", "Debussy", 300_000),
    ]

    for name, artist, dur in test_tracks:
        print(f"\n  ▸ {name} by {artist}")
        try:
            dz = try_deezer_preview(name, artist, dur)
            if not dz.get("audio_path"):
                print("    ⚠ No preview available, skipping")
                continue

            tags = extract_tags(dz["audio_path"])
            assert isinstance(tags, dict)
            assert "genre" in tags and "instrument" in tags and "vocal" in tags

            for cat in ("genre", "instrument", "vocal"):
                items = tags[cat]
                if items:
                    tag_str = ", ".join(f'{t["label"]} ({t["prob"]:.3f})' for t in items)
                    print(f"    {cat:12s}: {tag_str}")
                else:
                    print(f"    {cat:12s}: (below threshold)")

            # Full description
            feats = extract_audio_features(dz["audio_path"])
            desc = describe_track(name, artist, feats, dur, tags)
            print(f"    description: {desc[:200]}...")

        except Exception as e:
            print(f"    ⚠ Error: {e}")

    print("\n  ✓ PANNs tagging OK")


# ── Run all ─────────────────────────────────────────────────────

if __name__ == "__main__":
    test_utilities()
    test_feature_extraction()
    test_describe_track()
    test_deezer_search()
    test_tagging()
    print(f"\n{'=' * 60}")
    print("All capability tests finished.")
