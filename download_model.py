#!/usr/bin/env python
"""Download sd-turbo model with retry logic for unstable networks."""
import os
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download

# Base download settings.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

local_dir = Path("models/hf-cache/sd-turbo")
local_dir.parent.mkdir(parents=True, exist_ok=True)

print("Scaricando sd-turbo da Hugging Face Hub...")
print(f"Destinazione: {local_dir.resolve()}")

max_attempts = 8
for attempt in range(1, max_attempts + 1):
    try:
        print(f"Tentativo {attempt}/{max_attempts}...")
        path = snapshot_download(
            "stabilityai/sd-turbo",
            local_dir=str(local_dir),
            cache_dir="models/hf-cache",
            force_download=False,
            max_workers=2,
        )
        print(f"✓ Modello scaricato con successo in: {path}")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Tentativo {attempt} fallito: {e}")
        if attempt == max_attempts:
            sys.exit(1)
        wait_seconds = min(45, 5 * attempt)
        print(f"Attendo {wait_seconds}s e riprovo con resume automatico...")
        time.sleep(wait_seconds)
