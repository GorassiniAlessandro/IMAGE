from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SampleRecord:
    image_path: str
    caption: str
    width: int
    height: int
    blur_score: float
    perceptual_hash: str


@dataclass(frozen=True)
class FilterConfig:
    min_width: int = 384
    min_height: int = 384
    min_blur_score: float = 35.0
    duplicate_distance: int = 4
    fallback_caption: str = "photo of subject"


def discover_image_paths(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _normalize_key(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix().lower()
    except ValueError:
        return path.as_posix().lower()


def _caption_from_filename(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    return stem if stem else "photo of subject"


def load_caption_map(captions_path: str | Path | None, image_root: Path) -> dict[str, str]:
    if captions_path is None:
        return {}

    path = Path(captions_path)
    if not path.exists():
        return {}

    caption_map: dict[str, str] = {}

    if path.is_dir():
        for caption_file in path.rglob("*.txt"):
            caption = caption_file.read_text(encoding="utf-8").strip()
            if caption:
                caption_map[_normalize_key(caption_file.with_suffix(""), path)] = caption
        return caption_map

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            caption = (payload.get("caption") or payload.get("text") or "").strip()
            image_value = payload.get("image") or payload.get("image_path") or payload.get("path")
            if caption and image_value:
                caption_map[str(Path(image_value).as_posix()).lower()] = caption
                caption_map[Path(image_value).stem.lower()] = caption
        return caption_map

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, str) and value.strip():
                    caption_map[str(key).lower()] = value.strip()
        return caption_map

    # Plain text fallback: one caption per line, matched by image order later.
    return {}


def _get_caption_for_path(image_path: Path, image_root: Path, caption_map: dict[str, str], fallback_caption: str) -> str:
    relative_key = _normalize_key(image_path, image_root)
    if relative_key in caption_map:
        return caption_map[relative_key]
    stem_key = image_path.stem.lower()
    if stem_key in caption_map:
        return caption_map[stem_key]
    filename_key = image_path.name.lower()
    if filename_key in caption_map:
        return caption_map[filename_key]
    sidecar = image_path.with_suffix(".txt")
    if sidecar.exists():
        caption = sidecar.read_text(encoding="utf-8").strip()
        if caption:
            return caption
    return _caption_from_filename(image_path) or fallback_caption


def average_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    threshold = float(pixels.mean())
    bits = (pixels > threshold).astype(np.uint8).reshape(-1)
    packed = 0
    for bit in bits:
        packed = (packed << 1) | int(bit)
    return f"{packed:016x}"


def hamming_distance_hex(left: str, right: str) -> int:
    return int(int(left, 16) ^ int(right, 16)).bit_count()


def laplacian_variance(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(laplacian.var())


def inspect_image(path: Path) -> tuple[int, int, float, str]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        return width, height, laplacian_variance(rgb), average_hash(rgb)


def build_filtered_records(
    image_root: str | Path,
    captions_path: str | Path | None = None,
    config: FilterConfig | None = None,
) -> list[SampleRecord]:
    root = Path(image_root)
    config = config or FilterConfig()
    caption_map = load_caption_map(captions_path, root)

    candidates: list[SampleRecord] = []
    for image_path in discover_image_paths(root):
        try:
            width, height, blur_score, image_hash = inspect_image(image_path)
        except Exception:
            continue

        if width < config.min_width or height < config.min_height:
            continue
        if blur_score < config.min_blur_score:
            continue

        caption = _get_caption_for_path(image_path, root, caption_map, config.fallback_caption)
        candidates.append(
            SampleRecord(
                image_path=str(image_path),
                caption=caption,
                width=width,
                height=height,
                blur_score=blur_score,
                perceptual_hash=image_hash,
            )
        )

    candidates.sort(key=lambda item: (item.width * item.height, item.blur_score), reverse=True)

    filtered: list[SampleRecord] = []
    for candidate in candidates:
        duplicate = any(
            hamming_distance_hex(candidate.perceptual_hash, kept.perceptual_hash) <= config.duplicate_distance
            for kept in filtered
        )
        if not duplicate:
            filtered.append(candidate)

    return filtered


def save_manifest(records: Iterable[SampleRecord], manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return path


def load_manifest(manifest_path: str | Path) -> list[SampleRecord]:
    path = Path(manifest_path)
    records: list[SampleRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(SampleRecord(**payload))
    return records


def load_image_tensor(image_path: str | Path, resolution: int) -> torch.Tensor:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        short_edge = min(rgb.size)
        if short_edge == 0:
            raise ValueError(f"Immagine non valida: {image_path}")
        scale = resolution / short_edge
        resized = rgb.resize(
            (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
            Image.Resampling.LANCZOS,
        )
        left = max(0, (resized.width - resolution) // 2)
        top = max(0, (resized.height - resolution) // 2)
        cropped = resized.crop((left, top, left + resolution, top + resolution))
        array = np.asarray(cropped, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1)


class CaptionedImageDataset(Dataset):
    def __init__(
        self,
        image_root: str | Path,
        captions_path: str | Path | None = None,
        filter_config: FilterConfig | None = None,
        manifest_path: str | Path | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.captions_path = Path(captions_path) if captions_path else None
        self.filter_config = filter_config or FilterConfig()
        if manifest_path and Path(manifest_path).exists():
            self.records = load_manifest(manifest_path)
        else:
            self.records = build_filtered_records(self.image_root, self.captions_path, self.filter_config)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        return {
            "image_path": record.image_path,
            "caption": record.caption,
            "width": record.width,
            "height": record.height,
            "blur_score": record.blur_score,
            "perceptual_hash": record.perceptual_hash,
        }


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_dir: str | Path, preload_to_ram: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_files = sorted(self.cache_dir.glob("*.pt"))
        self.preload_to_ram = preload_to_ram and len(self.cache_files) <= 512
        self._memory_cache: list[dict[str, object]] | None = None
        if self.preload_to_ram:
            self._memory_cache = [torch.load(cache_file, map_location="cpu") for cache_file in self.cache_files]

    def __len__(self) -> int:
        return len(self.cache_files)

    def __getitem__(self, index: int) -> dict[str, object]:
        if self._memory_cache is not None:
            return self._memory_cache[index]
        return torch.load(self.cache_files[index], map_location="cpu")


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_cached_features if hasattr(dataset, "cache_files") else (lambda batch: batch),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


def collate_cached_features(batch: list[dict[str, object]]) -> dict[str, object]:
    latents = torch.stack([item["latents"] for item in batch], dim=0)
    prompt_embeds = torch.stack([item["prompt_embeds"] for item in batch], dim=0)
    captions = [str(item["caption"]) for item in batch]
    image_paths = [str(item["image_path"]) for item in batch]
    return {
        "latents": latents,
        "prompt_embeds": prompt_embeds,
        "captions": captions,
        "image_paths": image_paths,
    }


def stable_cache_name(record: SampleRecord) -> str:
    payload = f"{record.image_path}|{record.caption}|{record.perceptual_hash}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()
