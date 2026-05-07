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

# ROOT is the repository root from personal-lora/training/train.
ROOT = Path(__file__).resolve().parents[3]
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


DEFAULT_DATA_DIR = ROOT / "personal-lora" / "training" / "dataset" / "raw"
DEFAULT_CAPTIONS = ROOT / "personal-lora" / "training" / "dataset" / "captions.jsonl"
DEFAULT_EXPERIMENT_DIR = ROOT / "personal-lora"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "checkpoints"
DEFAULT_CACHE_DIR = DEFAULT_EXPERIMENT_DIR / "cache"
DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_LOG_DIR = DEFAULT_EXPERIMENT_DIR / "logs"
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
    train_text_encoder: bool = True


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


def prepare_cache(
    records,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    config: TrainConfig,
    logger: logging.Logger,
    cache_dir: Path,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_encoder.eval()
    vae.eval()
    processed = 0
    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            cache_name = stable_cache_name(record)
            cache_path = cache_dir / f"{cache_name}.pt"
            if cache_path.exists() and not config.force_rebuild_cache:
                continue

            pixel_values = load_image_tensor(record.image_path, config.resolution).unsqueeze(0)
            prompt_inputs = tokenizer(
                record.caption,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )

            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            prompt_embeds = text_encoder(prompt_inputs.input_ids)[0]

            payload = {
                "latents": latents.squeeze(0).to(dtype=torch.float32, device="cpu"),
                "prompt_embeds": prompt_embeds.squeeze(0).to(dtype=torch.float32, device="cpu"),
                "caption": record.caption,
                "image_path": record.image_path,
            }
            torch.save(payload, cache_path)
            processed += 1
            if processed % 20 == 0:
                logger.info("Cache created for %s images", processed)

    return cache_dir


def load_training_cache(cache_dir: Path, logger: logging.Logger) -> CachedFeatureDataset:
    dataset = CachedFeatureDataset(cache_dir, preload_to_ram=True)
    if dataset._memory_cache is not None:
        logger.info("Cache feature preloaded in RAM: %s items", len(dataset))
    else:
        logger.info("Cache feature read from disk on-demand: %s items", len(dataset))
    return dataset


def configure_lora_adapter(unet: UNet2DConditionModel, rank: int, alpha: int) -> None:
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights=True,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        inference_mode=False,
        bias="none",
    )
    unet.add_adapter(lora_config)
    unet.set_adapter("default")


def configure_text_encoder_lora(text_encoder: CLIPTextModel, rank: int, alpha: int) -> None:
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights=True,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
        bias="none",
    )
    text_encoder.add_adapter(lora_config)
    text_encoder.set_adapter("default")


def save_training_log(log_path: Path, payload: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_checkpoint(
    unet: UNet2DConditionModel,
    text_encoder: CLIPTextModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    output_dir: Path,
    step: int,
    epoch: int,
    loss_value: float,
    logger: logging.Logger,
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unet.save_lora_adapter(str(checkpoint_dir / "unet"), adapter_name="default", safe_serialization=True)
    # CLIPTextModel uses the PEFT save_pretrained API for adapter-only serialization.
    text_encoder.save_pretrained(
        str(checkpoint_dir / "text_encoder"),
        safe_serialization=True,
        selected_adapters=["default"],
    )
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    metadata = {
        "step": step,
        "epoch": epoch,
        "loss": loss_value,
    }
    (checkpoint_dir / "training_state.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Checkpoint saved in %s (unet + text_encoder)", checkpoint_dir)
    return checkpoint_dir


def train(
    config: TrainConfig,
    logger: logging.Logger,
) -> dict[str, object]:
    seed_everything(config.seed)
    
    # Resolve and load base model
    base_model_id, _ = resolve_base_model(config.base_model)
    logger.info("Loading base model: %s", base_model_id)
    
    pipeline = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32)
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    unet = pipeline.unet
    
    # Load dataset and manifest
    manifest_path = resolve_path(config.manifest_path, Path(config.manifest_path))
    if manifest_path.exists():
        records = load_manifest(manifest_path)
        logger.info("Dataset loaded from manifest: %s records", len(records))
    else:
        dataset = CaptionedImageDataset(
            config.data_dir,
            config.captions_file,
            FilterConfig(
                min_width=config.min_width,
                min_height=config.min_height,
                min_blur_score=config.min_blur_score,
                duplicate_distance=config.duplicate_distance,
            ),
        )
        records = dataset.records
        save_manifest(records, manifest_path)
        logger.info("Dataset filtered: %s images saved to %s", len(records), manifest_path)
    
    # Setup cache
    cache_dir = Path(config.cache_dir)
    if config.cache_latents and config.cache_text_embeddings:
        prepare_cache(records, tokenizer, text_encoder, vae, config, logger, cache_dir)
        train_dataset = load_training_cache(cache_dir, logger)
    else:
        train_dataset = CaptionedImageDataset(
            config.data_dir,
            config.captions_file,
            FilterConfig(
                min_width=config.min_width,
                min_height=config.min_height,
                min_blur_score=config.min_blur_score,
                duplicate_distance=config.duplicate_distance,
            ),
            manifest_path,
        )
    
    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    
    # Setup LoRA adapters
    configure_lora_adapter(unet, config.lora_rank, config.lora_alpha)
    if config.train_text_encoder:
        configure_text_encoder_lora(text_encoder, config.lora_rank, config.lora_alpha)
    
    # Get trainable parameters
    unet_trainable = [p for p in unet.parameters() if p.requires_grad]
    text_encoder_trainable = [p for p in text_encoder.parameters() if p.requires_grad] if config.train_text_encoder else []
    all_trainable = unet_trainable + text_encoder_trainable
    
    logger.info("UNet trainable params: %s, Text encoder trainable: %s", len(unet_trainable), len(text_encoder_trainable))
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(all_trainable, lr=config.learning_rate)
    
    # Setup scheduler
    max_steps = config.max_train_steps or (len(train_loader) * config.epochs)
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=max_steps,
    )
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(config.train_log_path)
    
    # Training loop
    unet.train()
    if config.train_text_encoder:
        text_encoder.train()
    
    global_step = 0
    epoch_losses = []
    
    for epoch in range(config.epochs):
        for batch_idx, batch in enumerate(train_loader):
            if config.max_train_steps and global_step >= config.max_train_steps:
                break
            
            latents = batch["latents"].to("cpu")
            prompt_embeds = batch["prompt_embeds"].to("cpu")
            
            with torch.no_grad():
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device="cpu")
                noisy_latents = latents + noise * math.sqrt(1 - (timesteps / 1000).view(-1, 1, 1, 1) ** 2)
            
            model_pred = unet(noisy_latents, timesteps, prompt_embeds).sample
            loss = F.mse_loss(model_pred, noise, reduction="mean") / config.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_losses.append(loss.item() * config.gradient_accumulation_steps)
                
                if global_step % config.log_every == 0:
                    log_entry = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": epoch_losses[-1],
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    save_training_log(log_path, log_entry)
                    logger.info("step=%d epoch=%d loss=%.6f lr=%.8f", global_step, epoch + 1, epoch_losses[-1], log_entry["lr"])
                
                if global_step % config.checkpoint_steps == 0:
                    save_checkpoint(unet, text_encoder, optimizer, lr_scheduler, output_dir, global_step, epoch + 1, epoch_losses[-1], logger)
    
    mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    logger.info("epoch=%d mean_loss=%.6f", epoch + 1, mean_epoch_loss)
    
    final_checkpoint = save_checkpoint(unet, text_encoder, optimizer, lr_scheduler, output_dir, global_step, config.epochs, mean_epoch_loss, logger)
    
    result = {
        "base_model": config.base_model,
        "dataset_size": len(records),
        "cache_size": len(train_dataset),
        "final_step": global_step,
        "final_loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "train_log_path": str(log_path),
        "text_encoder_trained": config.train_text_encoder,
    }
    
    logger.info("Training completed: %s", json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LoRA for Stable Diffusion (UNet + optional text encoder).")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--captions-file", default=str(DEFAULT_CAPTIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--train-log-path", default=str(DEFAULT_TRAIN_LOG))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--checkpoint-steps", type=int, default=200)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--lr-scheduler", default="constant_with_warmup")
    parser.add_argument("--lr-warmup-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--train-text-encoder", action="store_true", default=True)
    parser.add_argument("--no-train-text-encoder", action="store_false", dest="train_text_encoder")
    args = parser.parse_args()
    
    log_dir = Path(args.train_log_path).parent
    logger = configure_logging(log_dir)
    
    config = TrainConfig(
        base_model=args.base_model,
        data_dir=args.data_dir,
        captions_file=args.captions_file,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        manifest_path=args.manifest_path,
        train_log_path=args.train_log_path,
        resolution=args.resolution,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        checkpoint_steps=args.checkpoint_steps,
        max_train_steps=args.max_train_steps,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        log_every=args.log_every,
        train_text_encoder=args.train_text_encoder,
    )
    
    logger.info("Config loaded: %s", json.dumps(asdict(config), indent=2, ensure_ascii=False))
    train(config, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
