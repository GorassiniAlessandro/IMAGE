from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

# ROOT adjusted because file is now in training/val
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.evaluation.clip_score import ClipScoreResult, ClipScorer, write_clip_results_csv, write_clip_results_json


DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_CHECKPOINT_ROOT = ROOT / "training" / "output" / "personal-lora"
DEFAULT_OUTPUT_DIR = ROOT / "training" / "output" / "validation"
DEFAULT_BEST_DIR = ROOT / "models" / "lora" / "best"
DEFAULT_PROMPTS = [
    "portrait of pstyle subject, cinematic lighting",
    "portrait of pstyle subject, sharp focus, realistic skin texture",
    "pstyle subject in a detailed scene, balanced composition",
]


@dataclass(frozen=True)
class ValidationConfig:
    base_model: str = DEFAULT_BASE_MODEL
    checkpoint_dir: str | None = None
    checkpoint_root: str = str(DEFAULT_CHECKPOINT_ROOT)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    best_dir: str = str(DEFAULT_BEST_DIR)
    prompts: tuple[str, ...] = tuple(DEFAULT_PROMPTS)
    device: str = "cpu"
    steps: int = 12
    guidance_scale: float = 0.0
    width: int = 512
    height: int = 512
    images_per_prompt: int = 1
    seed: int = 42
    clip_model: str = "openai/clip-vit-base-patch32"
    lora_scale: float = 1.0


def resolve_path(value: str | None, default_path: Path) -> Path:
    if not value:
        return default_path
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def discover_checkpoints(root: Path) -> list[Path]:
    checkpoints = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")]
    checkpoints.sort(key=lambda item: int(item.name.split("-")[-1]))
    return checkpoints


def load_pipeline(base_model: str, device: str) -> StableDiffusionPipeline:
    dtype = torch.float32 if device == "cpu" else torch.float16
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    if device == "cpu":
        pipeline.enable_attention_slicing()
        pipeline.enable_vae_slicing()
    return pipeline


def generate_for_checkpoint(
    pipeline: StableDiffusionPipeline,
    checkpoint_dir: Path,
    prompts: list[str],
    output_dir: Path,
    config: ValidationConfig,
    clip_scorer: ClipScorer,
) -> list[ClipScoreResult]:
    checkpoint_output_dir = output_dir / checkpoint_dir.name
    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)

    pipeline.load_lora_weights(str(checkpoint_dir))
    if hasattr(pipeline, "fuse_lora"):
        pipeline.fuse_lora(lora_scale=config.lora_scale)

    results: list[ClipScoreResult] = []

    for prompt_index, prompt in enumerate(prompts):
        for image_index in range(config.images_per_prompt):
            seed_value = config.seed + prompt_index * 100 + image_index
            generator = torch.Generator(device=config.device).manual_seed(seed_value)
            rendered = pipeline(
                prompt=prompt,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
                width=config.width,
                height=config.height,
                generator=generator,
            )
            image = rendered.images[0]
            image_path = checkpoint_output_dir / f"{prompt_index + 1:02d}_{image_index + 1:02d}.png"
            image.save(image_path)
            score = clip_scorer.score_pair(image_path, prompt)
            results.append(
                ClipScoreResult(
                    checkpoint_name=checkpoint_dir.name,
                    image_path=str(image_path),
                    prompt=prompt,
                    score=score,
                )
            )

    return results


def copy_best_checkpoint(best_dir: Path, checkpoint_dir: Path) -> None:
    best_dir.parent.mkdir(parents=True, exist_ok=True)
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(checkpoint_dir, best_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Valuta checkpoint LoRA con prompt fissi e CLIP score.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--best-dir", default=str(DEFAULT_BEST_DIR))
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

    output_dir = resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = resolve_path(args.checkpoint_root, DEFAULT_CHECKPOINT_ROOT)
    best_dir = resolve_path(args.best_dir, DEFAULT_BEST_DIR)

    prompts = [prompt.strip() for prompt in (args.prompts or DEFAULT_PROMPTS) if prompt and prompt.strip()]
    if not prompts:
        raise ValueError("Nessun prompt di validazione valido fornito.")

    config = ValidationConfig(
        base_model=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_root=str(checkpoint_root),
        output_dir=str(output_dir),
        best_dir=str(best_dir),
        prompts=tuple(prompts),
        device=args.device,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        images_per_prompt=args.images_per_prompt,
        seed=args.seed,
        clip_model=args.clip_model,
        lora_scale=args.lora_scale,
    )

    checkpoint_dirs: list[Path]
    if args.checkpoint_dir:
        checkpoint_dirs = [resolve_path(args.checkpoint_dir, checkpoint_root)]
    else:
        checkpoint_dirs = discover_checkpoints(checkpoint_root)
        if not checkpoint_dirs:
            raise FileNotFoundError(f"Nessun checkpoint trovato in {checkpoint_root}")

    all_results: list[ClipScoreResult] = []
    best_checkpoint: Path | None = None
    best_score = float("-inf")
    clip_scorer = ClipScorer(model_name=config.clip_model, device=config.device)

    for checkpoint_dir in checkpoint_dirs:
        pipeline = load_pipeline(config.base_model, config.device)
        results = generate_for_checkpoint(pipeline, checkpoint_dir, prompts, output_dir, config, clip_scorer)
        all_results.extend(results)
        checkpoint_score = sum(result.score for result in results) / max(1, len(results))
        checkpoint_summary = {
            "checkpoint": checkpoint_dir.name,
            "score": checkpoint_score,
            "results": [result.__dict__ for result in results],
        }
        (output_dir / f"{checkpoint_dir.name}.json").write_text(
            json.dumps(checkpoint_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if checkpoint_score > best_score:
            best_score = checkpoint_score
            best_checkpoint = checkpoint_dir

    report_path_json = write_clip_results_json(all_results, output_dir / "clip_results.json")
    report_path_csv = write_clip_results_csv(all_results, output_dir / "clip_results.csv")

    summary = {
        "checkpoint_root": str(checkpoint_root),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "best_score": best_score,
        "results_json": str(report_path_json),
        "results_csv": str(report_path_csv),
        "count": len(all_results),
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if best_checkpoint is not None:
        copy_best_checkpoint(best_dir, best_checkpoint)
        summary["best_dir"] = str(best_dir)
        (output_dir / "validation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
