"""
ACE-Step music generation on Modal.
Replaces DiffRhythm with ACE-Step for higher quality output.
Based on: https://modal.com/docs/examples/generate_music

Deploy:   modal deploy modal/diffrhythm_app.py
Serve:    modal serve modal/diffrhythm_app.py
"""

import base64
import os
import traceback
from pathlib import Path
from uuid import uuid4

import modal

APP_NAME = "musicgen-ace-step"

cache_dir = "/root/.cache/ace-step/checkpoints"
model_cache = modal.Volume.from_name("ace-step-model-cache", create_if_missing=True)


def _download_model():
    """Pre-download model weights during image build so cold starts are fast."""
    from acestep.pipeline_ace_step import ACEStepPipeline

    ACEStepPipeline(dtype="bfloat16", cpu_offload=False, overlapped_decode=True)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "git+https://github.com/ace-step/ACE-Step.git",
        "fastapi[standard]",
        "hf_transfer",
    )
    .env({"HF_HUB_CACHE": cache_dir, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(_download_model, volumes={cache_dir: model_cache})
)

app = modal.App(APP_NAME)

_model = None


def _get_model():
    global _model
    if _model is None:
        token = os.environ.get("HF_TOKEN")
        if token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        from acestep.pipeline_ace_step import ACEStepPipeline

        _model = ACEStepPipeline(
            dtype="bfloat16", cpu_offload=False, overlapped_decode=True
        )
    return _model


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 10,
    volumes={cache_dir: model_cache},
    secrets=[modal.Secret.from_name("hf-token")],
)
@modal.fastapi_endpoint(method="POST")
def generate(data: dict):
    try:
        print(f"[generate] Received keys: {list(data.keys())}")

        text_prompt = (data.get("text_prompt") or "").strip()
        lyrics = (data.get("lyrics") or "").strip()
        if not text_prompt:
            return {"error": "text_prompt is required"}

        options = data.get("options") or {}
        duration = float(options.get("duration", 240.0))
        if duration < 30:
            duration = 30.0
        if duration > 240:
            duration = 240.0
        file_type = options.get("file_type", "mp3")
        seed = int(options.get("seed", 1))
        mode = (options.get("mode") or "quality").lower()

        # Defaults tuned closer to ACE-Step v1.5 UI expectations.
        if mode == "turbo":
            infer_step = int(options.get("infer_step", 8))
            guidance_scale = float(options.get("guidance_scale", 6.5))
        else:
            infer_step = int(options.get("infer_step", 32))
            guidance_scale = float(options.get("guidance_scale", 7.0))
        scheduler_type = options.get("scheduler_type", "euler")
        cfg_type = options.get("cfg_type", "apg")
        omega_scale = float(options.get("omega_scale", 10))
        guidance_interval = float(options.get("guidance_interval", 0.5))
        guidance_interval_decay = float(options.get("guidance_interval_decay", 0))
        min_guidance_scale = float(options.get("min_guidance_scale", 3))
        use_erg_tag = bool(options.get("use_erg_tag", True))
        use_erg_lyric = bool(options.get("use_erg_lyric", True))
        use_erg_diffusion = bool(options.get("use_erg_diffusion", True))

        if not lyrics:
            use_erg_lyric = False

        print(f"[generate] prompt={text_prompt[:80]!r}...")
        print(f"[generate] lyrics length={len(lyrics)}, duration={duration}s")

        model = _get_model()

        output_path = f"/dev/shm/output_{uuid4().hex}.{file_type}"

        model(
            audio_duration=duration,
            prompt=text_prompt,
            lyrics=lyrics,
            format=file_type,
            save_path=output_path,
            manual_seeds=seed,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=use_erg_tag,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
        )

        audio_bytes = Path(output_path).read_bytes()
        b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "file_type": file_type,
            "audio_base64": b64,
        }
    except Exception as e:
        print("[generate] error:", e)
        print(traceback.format_exc())
        return {"error": str(e)}
