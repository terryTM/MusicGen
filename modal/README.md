Modal DiffRhythm Host

Overview
- This app hosts DiffRhythm v1.2 full on Modal (A100).
- It exposes a single POST endpoint that returns base64 audio.

Prereqs
1. Create a Hugging Face token and accept the DiffRhythm-VAE license.
2. Store the token as a Modal secret named `hf-token` with key `HF_TOKEN`.

Create Modal secret
1. Create a secret in Modal with key `HF_TOKEN`.
2. Name the secret `hf-token`.

Deploy
1. Install Modal CLI and authenticate.
2. Deploy:
   modal deploy modal/diffrhythm_app.py

Use
- The deployed endpoint will look like:
  https://<your-username>--diffrhythm-host-generate.modal.run

Request shape
POST JSON:
{
  "text_prompt": "Generate a rap song with heavy drums and a dark bassline.",
  "lyrics": "[00:05.00] ...",
  "options": {
    "duration": 95,
    "steps": 32,
    "cfg_strength": 4.0,
    "odeint_method": "euler",
    "preference": "quality first",
    "file_type": "mp3",
    "seed": 0,
    "randomize_seed": true
  }
}

Response
{
  "file_type": "mp3",
  "audio_base64": "<base64>"
}

Notes
- The first request will be slow while weights download.
- A100 is recommended. If you need a different GPU, change `gpu="A100"` in modal/diffrhythm_app.py.
