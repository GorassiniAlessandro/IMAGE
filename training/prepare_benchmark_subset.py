from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dataset import FilterConfig, build_filtered_records


DEFAULT_SOURCE_DIR = ROOT / "training" / "dataset" / "raw"
DEFAULT_OUTPUT_DIR = ROOT / "training" / "dataset" / "benchmark"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "benchmark_manifest.jsonl"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "benchmark_summary.json"
DEFAULT_FILTER_CONFIG = FilterConfig()


def _resolve_path(value: str | None, default_path: Path) -> Path:
    if not value:
        return default_path
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def _copy_sidecar_assets(image_path: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    sidecar_json = image_path.with_suffix(".json")
    sidecar_txt = image_path.with_suffix(".txt")
    if sidecar_json.exists():
        shutil.copy2(sidecar_json, destination_dir / sidecar_json.name)
    if sidecar_txt.exists():
        shutil.copy2(sidecar_txt, destination_dir / sidecar_txt.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Crea un subset filtrato e ripetibile per benchmark CPU.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--captions-file", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--min-width", type=int, default=384)
    parser.add_argument("--min-height", type=int, default=384)
    parser.add_argument("--min-blur-score", type=float, default=35.0)
    parser.add_argument("--duplicate-distance", type=int, default=4)
    parser.add_argument("--fallback-caption", default="photo of subject")
    parser.add_argument("--copy-assets", action="store_true", default=True)
    parser.add_argument("--no-copy-assets", action="store_false", dest="copy_assets")
    args = parser.parse_args()

    source_dir = _resolve_path(args.source_dir, DEFAULT_SOURCE_DIR)
    output_dir = _resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    manifest_path = _resolve_path(args.manifest_path, DEFAULT_MANIFEST_PATH)
    summary_path = _resolve_path(args.summary_path, DEFAULT_SUMMARY_PATH)
    captions_file = _resolve_path(args.captions_file, source_dir.parent / "captions.jsonl") if args.captions_file else None

    records = build_filtered_records(
        image_root=source_dir,
        captions_path=captions_file,
        config=FilterConfig(
            min_width=args.min_width,
            min_height=args.min_height,
            min_blur_score=args.min_blur_score,
            duplicate_distance=args.duplicate_distance,
            fallback_caption=args.fallback_caption,
        ),
    )

    if not records:
        print(f"Nessuna immagine valida trovata in {source_dir}")
        return 1

    selected = records[: max(1, args.limit)]
    benchmark_raw_dir = output_dir / "raw"
    benchmark_raw_dir.mkdir(parents=True, exist_ok=True)

    benchmark_records = []
    for record in selected:
        source_image = Path(record.image_path)
        target_image = benchmark_raw_dir / source_image.name
        shutil.copy2(source_image, target_image)
        if args.copy_assets:
            _copy_sidecar_assets(source_image, benchmark_raw_dir)
        benchmark_records.append(
            {
                "image_path": str(target_image),
                "caption": record.caption,
                "width": record.width,
                "height": record.height,
                "blur_score": record.blur_score,
                "perceptual_hash": record.perceptual_hash,
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in benchmark_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "count": len(benchmark_records),
        "selection_limit": max(1, args.limit),
        "copy_assets": args.copy_assets,
        "filters": {
            "min_width": args.min_width,
            "min_height": args.min_height,
            "min_blur_score": args.min_blur_score,
            "duplicate_distance": args.duplicate_distance,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
