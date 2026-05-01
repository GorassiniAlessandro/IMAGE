from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "training" / "dataset" / "raw"


def _save_sample(image, prompt: str, source: str, source_row: int, config_name: str) -> None:
    base = _safe_name("open", f"{source}_{config_name}_{source_row}_{prompt}")
    img_path = OUT_DIR / f"{base}.png"
    meta_path = OUT_DIR / f"{base}.json"

    image.save(img_path, format="PNG", optimize=True)
    meta = {
        "image": img_path.name,
        "caption": prompt,
        "prompt": prompt,
        "negative_prompt": "",
        "style": "OpenDataset",
        "aspect_ratio": "unknown",
        "seed": None,
        "source": source,
        "config": config_name,
        "row_index": source_row,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_name(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _build_from_diffusiondb(sample_count: int, seed: int, config_name: str) -> int:
    from datasets import load_dataset

    print(f"Carico dataset poloclub/diffusiondb ({config_name})...")
    ds = load_dataset("poloclub/diffusiondb", config_name, split="train")

    total = len(ds)
    take = min(sample_count, total)
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    selected = indices[:take]

    saved = 0
    for idx in selected:
        row = ds[idx]
        image = row.get("image")
        prompt = (row.get("prompt") or "").strip()
        if image is None or not prompt:
            continue
        _save_sample(image=image, prompt=prompt, source="poloclub/diffusiondb", source_row=idx, config_name=config_name)
        saved += 1
    return saved


def _build_from_cifar10(sample_count: int, seed: int, config_name: str) -> int:
    from torchvision.datasets import CIFAR10

    print("Carico dataset CIFAR-10 (open/public, no login)...")
    data_root = ROOT / "training" / "dataset" / "cifar10-cache"
    ds = CIFAR10(root=str(data_root), train=True, download=True)
    class_names = ds.classes

    total = len(ds)
    take = min(sample_count, total)
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    selected = indices[:take]

    saved = 0
    for idx in selected:
        image, label_idx = ds[idx]
        label_name = class_names[label_idx]
        prompt = f"a detailed photo of a {label_name}, high quality"
        _save_sample(image=image, prompt=prompt, source="cifar10", source_row=idx, config_name=config_name)
        saved += 1
    return saved


def _build_synthetic(sample_count: int, seed: int, config_name: str) -> int:
    print("Creo dataset sintetico locale (offline, no download)...")
    rng = random.Random(seed)
    subjects = [
        "rock",
        "forest",
        "city skyline",
        "portrait",
        "spaceship",
        "castle",
        "dragon",
        "sports car",
    ]
    styles = [
        "cinematic",
        "anime",
        "photographic",
        "concept art",
        "fantasy",
        "watercolor",
        "noir",
    ]

    saved = 0
    for idx in range(sample_count):
        width = 512
        height = 512
        bg = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        image = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(image)

        for _ in range(8):
            x1, y1 = rng.randint(0, width - 1), rng.randint(0, height - 1)
            x2, y2 = rng.randint(0, width - 1), rng.randint(0, height - 1)
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            draw.rectangle([left, top, right, bottom], outline=color, width=2)

        subject = rng.choice(subjects)
        style = rng.choice(styles)
        prompt = f"{style} artwork of a {subject}, high detail"
        _save_sample(image=image, prompt=prompt, source="synthetic", source_row=idx, config_name=config_name)
        saved += 1
    return saved


def build_dataset(sample_count: int, seed: int, source: str, config_name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if source == "diffusiondb":
        saved = _build_from_diffusiondb(sample_count=sample_count, seed=seed, config_name=config_name)
    elif source == "synthetic":
        saved = _build_synthetic(sample_count=sample_count, seed=seed, config_name=config_name)
    else:
        saved = _build_from_cifar10(sample_count=sample_count, seed=seed, config_name=config_name)

    print(f"Completato. Salvate {saved} immagini in: {OUT_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scarica un dataset aperto da Hugging Face e prepara immagini per training LoRA."
    )
    parser.add_argument("--sample-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source",
        default="synthetic",
        choices=["synthetic", "cifar10", "diffusiondb"],
        help="Sorgente dataset: synthetic (offline), cifar10 o diffusiondb.",
    )
    parser.add_argument(
        "--config",
        default="2m_first_1k",
        help="Configurazione DiffusionDB (es: 2m_first_1k, large_random_1k)",
    )
    args = parser.parse_args()

    build_dataset(
        sample_count=max(1, args.sample_count),
        seed=args.seed,
        source=args.source,
        config_name=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
