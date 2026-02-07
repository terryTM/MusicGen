"""
api.py – Flask microservice wrapping the audio-analysis pipeline.
=================================================================
Endpoints
---------
POST /analyze          – Analyze a single track (Deezer search + features + tags + description)
POST /analyze-batch    – Analyze a list of tracks in one call
GET  /health           – Liveness check

The Node.js frontend server proxies requests here.
Start with:  conda run -n musicgen python models/api.py
"""

import os, sys, json, traceback
from flask import Flask, request, jsonify

# Ensure the models package is importable when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from ytm_to_deezer import (
    try_deezer_preview,
    extract_audio_features,
    extract_tags,
    describe_track,
)

app = Flask(__name__)


def _analyze_one(
    name: str,
    artist: str,
    duration_ms: int | None,
    preview_url: str | None = None,
) -> dict:
    """Run the full pipeline for a single track.

    If *preview_url* is supplied (e.g. the Node frontend already found it
    via its own Deezer search), we download and analyse that directly
    instead of re-searching Deezer with the raw YouTube title.
    """
    result = {
        "name": name,
        "artist": artist,
        "duration_ms": duration_ms,
        "deezer": None,
        "audio_features": None,
        "tags": None,
        "description": None,
    }

    audio_path = None

    # Fast path: Node already resolved a Deezer preview URL
    if preview_url:
        import hashlib
        h = hashlib.sha1(preview_url.encode("utf-8")).hexdigest()[:12]
        audio_path = os.path.join(
            os.path.dirname(__file__), "cache_deezer_previews", f"node_{h}.mp3"
        )
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        try:
            from ytm_to_deezer import download_file
            download_file(preview_url, audio_path)
        except Exception as e:
            print(f"[api] preview download failed for {name}: {e}")
            audio_path = None

    # Slow path: search Deezer ourselves
    if audio_path is None:
        dz = try_deezer_preview(name, artist, duration_ms)
        result["deezer"] = dz.get("deezer")
        if dz.get("found"):
            audio_path = dz["audio_path"]

    if audio_path and os.path.exists(audio_path):
        result["audio_features"] = extract_audio_features(audio_path)
        result["tags"] = extract_tags(audio_path)

    result["description"] = describe_track(
        name=name,
        artist=artist,
        audio_feats=result["audio_features"],
        duration_ms=duration_ms,
        tags=result["tags"],
    )
    return result


# ── Endpoints ────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze a single track.

    JSON body: { name, artist, duration_ms? }
    """
    body = request.get_json(force=True)
    name = body.get("name", "")
    artist = body.get("artist", "")
    duration_ms = body.get("duration_ms")
    preview_url = body.get("preview_url")

    if not name:
        return jsonify({"error": "name is required"}), 400

    try:
        result = _analyze_one(name, artist, duration_ms, preview_url=preview_url)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-batch", methods=["POST"])
def analyze_batch():
    """Analyze multiple tracks.

    JSON body: { tracks: [{ name, artist, duration_ms? }, ...] }
    """
    body = request.get_json(force=True)
    tracks = body.get("tracks", [])

    if not tracks:
        return jsonify({"error": "tracks list is required"}), 400

    results = []
    for t in tracks:
        try:
            r = _analyze_one(
                t.get("name", ""),
                t.get("artist", ""),
                t.get("duration_ms"),
                preview_url=t.get("preview_url"),
            )
            results.append(r)
        except Exception as e:
            traceback.print_exc()
            results.append({
                "name": t.get("name"),
                "artist": t.get("artist"),
                "error": str(e),
            })

    return jsonify({"results": results})


if __name__ == "__main__":
    print("🔬 Audio-analysis API starting on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
