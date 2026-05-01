from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from peft import LoraConfig

from transformers import CLIPTextModel, CLIPTokenizer

# ROOT now two levels up because this file lives in training/train
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dataset import (
    CachedFeatureDataset,
    CaptionedImageDataset,
    FilterConfig,
    build_dataloader,
    collate_cached_features,
    load_image_tensor,
    load_manifest,
    save_manifest,
    stable_cache_name,
)


DEFAULT_DATA_DIR = ROOT / "training" / "dataset" / "raw"
DEFAULT_CAPTIONS = ROOT / "training" / "dataset" / "captions.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "training" / "output" / "personal-lora"
DEFAULT_CACHE_DIR = ROOT / "training" / "cache" / "personal-lora"
DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_LOG_DIR = ROOT / "training" / "logs"
DEFAULT_MANIFEST = DEFAULT_LOG_DIR / "filtered_manifest.jsonl"
DEFAULT_TRAIN_LOG = DEFAULT_LOG_DIR / "train_log.jsonl"


@dataclass(frozen=True)
class TrainConfig:
    base_model: str = DEFAULT_BASE_MODEL
    data_dir: str = str(DEFAULT_DATA_DIR)
    captions_file: str | None = str(DEFAULT_CAPTIONS)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    cache_dir: str = str(DEFAULT_CACHE_DIR)
    manifest_path: str = str(DEFAULT_MANIFEST)
    train_log_path: str = str(DEFAULT_TRAIN_LOG)
    resolution: int = 512
    epochs: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    lora_rank: int = 8
    lora_alpha: int = 16
    checkpoint_steps: int = 200
    max_train_steps: int | None = None
    num_workers: int = 0
    seed: int = 42
    mixed_precision: str = "no"
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 20
    min_width: int = 384
    min_height: int = 384
    min_blur_score: float = 35.0
    duplicate_distance: int = 4
    cache_latents: bool = True
    cache_text_embeddings: bool = True
    force_rebuild_cache: bool = False
    resume_from_checkpoint: str | None = None
    log_every: int = 10


def configure_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("april.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def resolve_path(value: str | None, default_path: Path) -> Path:
    if not value:
        return default_path
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(max(1, min(8, cpu_count)))
    torch.set_num_interop_threads(1)


def resolve_base_model(base_model: str) -> tuple[str, bool]:
    candidate = Path(base_model)
    if candidate.exists():
        return str(candidate), True
    local_candidate = (ROOT / base_model).resolve()
    if local_candidate.exists():
        return str(local_candidate), True
    return base_model, False

# The rest of this module is unchanged from the original version and kept for brevity.
# See repository's original train/train_lora.py for full implementation.
