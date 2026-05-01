from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "training" / "dataset" / "raw"
DEFAULT_OUTPUT_DIR = ROOT / "training" / "output" / "personal-lora"


TRAINING_PROFILE_DEFAULTS = {
    "quick": {
        "resolution": 384,
        "epochs": 5,
        "learning_rate": 1e-5,
        "lora_rank": 8,
        "lora_alpha": 16,
        "checkpoint_steps": 100,
        "gradient_accumulation_steps": 4,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 10,
    },
    "balanced": {
        "resolution": 512,
        "epochs": 10,
        "learning_rate": 1e-5,
        "lora_rank": 8,
        "lora_alpha": 16,
        "checkpoint_steps": 150,
        "gradient_accumulation_steps": 4,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 20,
    },
    "quality": {
        "resolution": 512,
        "epochs": 15,
        "learning_rate": 1e-5,
        "lora_rank": 16,
        "lora_alpha": 32,
        "checkpoint_steps": 200,
        "gradient_accumulation_steps": 4,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 30,
    },
}


def _count_images(path: Path) -> int:
    return sum(1 for extension in ("*.png", "*.jpg", "*.jpeg", "*.webp") for _ in path.glob(extension))


def _build_command(args: argparse.Namespace) -> list[str]:
    defaults = TRAINING_PROFILE_DEFAULTS[args.training_profile]
    base_model = args.base_model_path if args.base_model_path else args.base_model
    command = [
        sys.executable,
        str(ROOT / "training" / "train" / "train_lora.py"),
        "--base-model",
        base_model,
        "--data-dir",
        str(args.data_dir),
        "--captions-file",
        str(args.captions_file),
        "--output-dir",
        str(args.output_dir),
        "--cache-dir",
        str(args.cache_dir),
        "--resolution",
        str(args.resolution or defaults["resolution"]),
        "--epochs",
        str(args.epochs or defaults["epochs"]),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps or defaults["gradient_accumulation_steps"]),
        "--learning-rate",
        str(args.learning_rate or defaults["learning_rate"]),
        "--lora-rank",
        str(args.lora_rank or defaults["lora_rank"]),
        "--lora-alpha",
        str(args.lora_alpha or defaults["lora_alpha"]),
        "--checkpoint-steps",
        str(args.checkpoint_steps or defaults["checkpoint_steps"]),
        "--lr-scheduler",
        args.lr_scheduler or defaults["lr_scheduler"],
        "--lr-warmup-steps",
        str(args.lr_warmup_steps or defaults["lr_warmup_steps"]),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
    ]

    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        if manifest_path.exists():
            command.extend(["--manifest-path", str(manifest_path)])

    if args.max_train_steps is not None:
        command.extend(["--max-train-steps", str(args.max_train_steps)])

    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-friendly LoRA training wrapper for APRIL.")
    parser.add_argument("--training-profile", choices=tuple(TRAINING_PROFILE_DEFAULTS), default="quality")
    parser.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--captions-file", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=str(ROOT / "training" / "cache" / "personal-lora"))
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--checkpoint-steps", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--lr-scheduler", default=None)
    parser.add_argument("--lr-warmup-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    captions_file = Path(args.captions_file) if args.captions_file else data_dir.parent / "captions.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_count = _count_images(data_dir)
    if image_count < 12:
        print(f"Dataset insufficiente: trovate {image_count} immagini in {data_dir}")
        return 1

    cmd = _build_command(
        argparse.Namespace(
            training_profile=args.training_profile,
            base_model=args.base_model,
            base_model_path=args.base_model_path,
            data_dir=data_dir,
            captions_file=captions_file,
            output_dir=output_dir,
            cache_dir=cache_dir,
            manifest_path=Path(args.manifest_path) if args.manifest_path else None,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            resolution=args.resolution,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            checkpoint_steps=args.checkpoint_steps,
            max_train_steps=args.max_train_steps,
            lr_scheduler=args.lr_scheduler,
            lr_warmup_steps=args.lr_warmup_steps,
            seed=args.seed,
            num_workers=args.num_workers,
        )
    )

    cmd_preview = " ".join(cmd)
    preview_path = output_dir / "last_train_command.txt"
    preview_path.write_text(cmd_preview, encoding="utf-8")

    print("Comando training pronto:")
    print(cmd_preview)
    print(f"Salvato anche in: {preview_path}")

    if args.dry_run:
        return 0

    print("Avvio training LoRA separato dalla validazione...")
    completed = subprocess.run(cmd, cwd=ROOT)
    result = {
        "exit_code": completed.returncode,
        "dataset_images": image_count,
        "output_dir": str(output_dir),
    }
    (output_dir / "last_train_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
