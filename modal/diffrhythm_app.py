import base64
import os
import random
import tempfile
from typing import Optional

import modal

APP_NAME = "diffrhythm-host"
DIFFRHYTHM_REPO = "https://huggingface.co/spaces/ASLP-lab/DiffRhythm"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "git-lfs", "ffmpeg", "espeak-ng",
        "build-essential", "cmake", "make", "gcc", "g++",
    )
    .env({"CC": "gcc", "CXX": "g++"})
    .run_commands(
        "git lfs install",
        f"git clone {DIFFRHYTHM_REPO} /opt/diffrhythm",
        "pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 "
        "torch==2.6.0+cu124 torchaudio==2.6.0+cu124",
        "pip install --no-cache-dir -r /opt/diffrhythm/requirements.txt",
        "pip install --no-cache-dir fastapi[standard] pydub",
    )
    .env({"PYTHONPATH": "/opt/diffrhythm"})
)

app = modal.App(APP_NAME)

MODEL_VOL = modal.Volume.from_name("diffrhythm-models", create_if_missing=True)

_MODEL_CACHE = {}


def _ensure_hf_env():
    token = os.environ.get("HF_TOKEN")
    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HOME", "/models/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/models/hf")


def _load_models(max_frames: int, device: str):
    key = (max_frames, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    os.chdir("/opt/diffrhythm")
    _ensure_hf_env()

    from diffrhythm.infer.infer_utils import prepare_model

    cfm, tokenizer, muq, vae, eval_model, eval_muq = prepare_model(max_frames, device)
    _MODEL_CACHE[key] = (cfm, tokenizer, muq, vae, eval_model, eval_muq)
    return _MODEL_CACHE[key]


def _run_infer(
    lyrics: str,
    text_prompt: str,
    duration: int = 95,
    steps: int = 32,
    cfg_strength: float = 4.0,
    odeint_method: str = "euler",
    preference: str = "quality first",
    file_type: str = "mp3",
    seed: int = 0,
    randomize_seed: bool = True,
):
    import torch
    from diffrhythm.infer.infer import inference
    from diffrhythm.infer.infer_utils import (
        get_lrc_token,
        get_negative_style_prompt,
        get_reference_latent,
        get_text_style_prompt,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_frames = 2048 if duration == 95 else 6144
    sway_sampling_coef = -1 if steps < 32 else None

    if randomize_seed:
        seed = random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)

    cfm, tokenizer, muq, vae, eval_model, eval_muq = _load_models(max_frames, device)

    lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
        max_frames, lyrics, tokenizer, duration, device
    )
    style_prompt = get_text_style_prompt(muq, text_prompt)
    negative_style_prompt = get_negative_style_prompt(device)
    latent_prompt, pred_frames = get_reference_latent(
        device, max_frames, False, None, None, vae
    )

    batch_infer_num = 5 if preference == "quality first" else 1

    output = inference(
        cfm_model=cfm,
        vae_model=vae,
        eval_model=eval_model,
        eval_muq=eval_muq,
        cond=latent_prompt,
        text=lrc_prompt,
        duration=end_frame,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        steps=steps,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        start_time=start_time,
        file_type=file_type,
        vocal_flag=False,
        odeint_method=odeint_method,
        pred_frames=pred_frames,
        batch_infer_num=batch_infer_num,
        song_duration=song_duration,
    )

    if file_type == "wav":
        # output is (sample_rate, np.ndarray)
        import io
        import soundfile as sf

        sr, audio_np = output
        buf = io.BytesIO()
        sf.write(buf, audio_np, sr, format="WAV")
        return buf.getvalue(), "wav"

    # mp3/ogg returns raw bytes
    return output, file_type


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    volumes={"/models": MODEL_VOL},
    secrets=[modal.Secret.from_name("hf-token")],
)
@modal.fastapi_endpoint(method="POST")
def generate(data: dict):
    lyrics = (data.get("lyrics") or "").strip()
    text_prompt = (data.get("text_prompt") or "").strip()
    if not text_prompt:
        return {"error": "text_prompt is required"}

    options = data.get("options") or {}
    duration = int(options.get("duration", 95))
    steps = int(options.get("steps", 32))
    cfg_strength = float(options.get("cfg_strength", 4.0))
    odeint_method = options.get("odeint_method", "euler")
    preference = options.get("preference", "quality first")
    file_type = options.get("file_type", "mp3")
    seed = int(options.get("seed", 0))
    randomize_seed = bool(options.get("randomize_seed", True))

    audio_bytes, out_type = _run_infer(
        lyrics=lyrics,
        text_prompt=text_prompt,
        duration=duration,
        steps=steps,
        cfg_strength=cfg_strength,
        odeint_method=odeint_method,
        preference=preference,
        file_type=file_type,
        seed=seed,
        randomize_seed=randomize_seed,
    )

    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return {
        "file_type": out_type,
        "audio_base64": b64,
    }
