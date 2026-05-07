# Personal Image Studio

Personal Image Studio is a local, customizable image-generation base that can also be driven by external agents. It does not copy Fooocus; instead, it keeps the same practical spirit and turns it into an original, modular architecture that you control.

The design goal is simple:

- you use the app locally through your own UI
- another AI or a script can drive it through the API
- you can change provider, profiles, and behavior without rewriting the whole stack

## What It Does

- generates preview images immediately, even without a GPU or heavy models
- exposes a simple local web UI
- stores configuration in a persistent local file
- lets you create reusable profiles
- exposes REST endpoints for automation and future integrations
- keeps the image-generation provider separate from the rest of the app

## Architecture

The project is organized in three layers:

1. Local web UI: a simple page for prompts, profiles, and outputs.
2. FastAPI API: endpoints for image generation, configuration, and profile management.
3. Image provider: a swappable component that currently produces SVG previews, but can later connect to Fooocus or another engine.

## Requirements

- Windows, Linux, or macOS
- Python 3.10 or newer
- a Python virtual environment is recommended
- optionally, a GPU and a real image backend for the next stage

## Project Layout

- `src/personal_image_studio/app.py`: FastAPI app, UI, and provider implementation
- `personal-lora/training/dataset.py`: dataset filtering, caption loading, and manifest creation
- `personal-lora/training/`: training code, dataset helpers, and dataset assets for the current LoRA workspace
- `personal-lora/`: named workspace for the current LoRA experiment, logs, caches, checkpoints, validation outputs, and training dataset
- `personal-lora/training/train/train_lora.py`: main LoRA training entry point
- `personal-lora/training/train_personal_lora.py`: training wrapper with profile presets
- `personal-lora/training/val/validate.py`: offline validation and CLIP scoring
- `personal-lora/training/evaluation/clip_score.py`: CLIP scoring helpers
- `pyproject.toml`: package dependencies and configuration
- `studio_config.json`: local runtime configuration
- `README.md`: project documentation

## Quick Start

1. Open the project folder.
2. Activate your Python virtual environment if you want to use the existing one.
3. Install the project in editable mode:

```bash
pip install -e .
```

4. Start the server:

```bash
uvicorn personal_image_studio.app:app --app-dir src --reload
```

5. Open the browser at:

```text
http://127.0.0.1:8000
```

## How To Use It

In the UI you can:

- write a prompt
- choose a profile
- set style, aspect ratio, and image count
- add a creative note for external automations
- generate immediate previews

The current previews are SVG placeholders. They are meant to test the end-to-end flow without depending on heavy models yet.

## Custom Profiles

Profiles are reusable presets. The project ships with a few base profiles:

- default
- cinematic
- portrait
- batch

You can edit them through the API or directly in `studio_config.json`.

Each profile can control:

- display name
- default style
- default aspect ratio
- default image count
- notes

## Local Configuration

Persistent configuration is stored in `studio_config.json` at the project root. If the file does not exist, the app starts with defaults and creates it when needed.

Main fields:

- `provider`: active provider, for example `mock` or `fooocus`
- `fooocus_endpoint`: Fooocus service address if you connect one later
- `local_model_id`: base model used by the local generator
- `local_use_personal_lora`: enables the trained personal LoRA
- `local_personal_lora_path`: path to the LoRA weights used by the app
- `profiles`: map of custom profiles

## Environment Variables

You can override the project behavior with these variables:

- `IMAGE_AI_PROVIDER`: forces the active provider, for example `mock` or `fooocus`
- `FOOOCUS_ENDPOINT`: local or remote Fooocus backend endpoint

## API Endpoints

The project is designed to be controlled by another AI or by scripts. The main endpoints are:

- `GET /api/health`: service status and active provider
- `GET /api/capabilities`: list of available capabilities
- `GET /api/config`: current configuration
- `PUT /api/config`: update configuration and profiles
- `GET /api/profiles`: list available profiles
- `POST /api/generate`: generate images from a prompt

### Generation Request Example

```json
{
  "prompt": "cinematic portrait of an android in a city at sunset",
  "negative_prompt": "blurry, low quality, extra fingers",
  "profile": "cinematic",
  "style": "Default",
  "aspect_ratio": "1024x1024",
  "count": 1,
  "seed": null,
  "creative_note": "keep a cool palette and neon lights"
}
```

### Response Example

```json
{
  "provider": "mock",
  "profile": "cinematic",
  "items": [
    {
      "title": "Preview 1",
      "image_data_uri": "data:image/svg+xml;base64,...",
      "prompt": "cinematic portrait of an android in a city at sunset",
      "notes": "Profile: cinematic | Style: Cinematic | Aspect ratio: 1344x768 | Creative note: keep a cool palette and neon lights"
    }
  ]
}
```

## Training Pipeline

Training has been split into a dedicated `personal-lora/training/` area so the app remains clean and the training workflow is easier to maintain.

The main pieces are:

- `personal-lora/training/dataset.py`: filters images, loads captions, creates manifests, and builds dataloaders
- `personal-lora/training/prepare_benchmark_subset.py`: creates a reproducible filtered benchmark subset
- `personal-lora/training/train/train_lora.py`: main LoRA training script
- `personal-lora/training/train_personal_lora.py`: wrapper that selects a profile and launches training
- `personal-lora/training/val/validate.py`: runs offline validation on saved checkpoints
- `personal-lora/training/evaluation/clip_score.py`: computes CLIP similarity scores for generated outputs

Current training defaults are tuned for CPU-friendly runs and keep validation outside the training loop.

## Training Output

The current training output path used by the app is:

```text
personal-lora/checkpoints/pytorch_lora_weights.safetensors
```

The app loads the personal LoRA from this path when `local_use_personal_lora` is enabled.

## Working With Another AI Or Automation

The project is designed to work well with an external agent. Another AI can:

- read `GET /api/capabilities`
- read profiles with `GET /api/profiles`
- update profiles with `PUT /api/config`
- send prompts and creative notes with `POST /api/generate`

This makes the project useful both for manual use and for automated workflows.

## Current Status

The project is already usable as a local base and as a demonstrator API. The real image backend can be connected later without changing the overall structure.

## License

GPL-3.0-only.
