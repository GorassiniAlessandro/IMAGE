from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Wrapper per la validazione separata dei checkpoint LoRA.")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--best-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--images-per-prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "val" / "validate.py"),
        "--device",
        args.device,
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--images-per-prompt",
        str(args.images_per_prompt),
        "--seed",
        str(args.seed),
        "--clip-model",
        args.clip_model,
        "--lora-scale",
        str(args.lora_scale),
    ]
    if args.checkpoint_dir:
        cmd.extend(["--checkpoint-dir", args.checkpoint_dir])
    if args.checkpoint_root:
        cmd.extend(["--checkpoint-root", args.checkpoint_root])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.best_dir:
        cmd.extend(["--best-dir", args.best_dir])
    if args.prompts:
        cmd.extend(["--prompts", *args.prompts])

    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())