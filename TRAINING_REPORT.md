# APRIL Diffusion Training Report

## Goal

Build a high-quality but computation-efficient LoRA training pipeline for Stable Diffusion on CPU-only hardware.

The design below is optimized for:
- small, high-quality datasets
- reproducible LoRA fine-tuning
- no validation inside the training loop
- separate offline evaluation with CLIP scoring
- easy integration into the APRIL system

## Recommended Training Strategy

Use Stable Diffusion 1.5 as the base model instead of a turbo variant for training.

Why:
- SD 1.5 is better documented and more stable for LoRA fine-tuning.
- Turbo models are optimized for speed at inference, not necessarily for training quality.
- On CPU, the bottleneck is the training loop itself, so we should maximize training stability and minimize waste.

The main rule is simple:
- train only on curated data
- save checkpoints periodically
- do not generate validation images during training
- evaluate later in a separate script

## Proposed Project Layout

```text
/data
  raw/
  filtered/
  captions/
/train
  dataset.py
  train_lora.py
/val
  validate.py
/models
  base_model/
  lora/
/training
  configs/
  logs/
/evaluation
  clip_score.py
```

Recommended module responsibilities:
- `train/dataset.py`: filtering, caption loading, dataset class, dataloader factory
- `train/train_lora.py`: LoRA fine-tuning entry point
- `val/validate.py`: load a checkpoint and generate images from fixed prompts
- `evaluation/clip_score.py`: compute CLIP similarity and log scores over time

## Dataset Handling

### Target dataset size

Use about 50 to 120 high-quality images.

If you currently have around 150 images, trim aggressively:
- remove duplicates
- remove blurry, dark, low-detail, or inconsistent images
- remove images with bad framing, weird artifacts, or wrong identity/style
- keep only images that support the intended visual concept

### Caption quality

Captions must be:
- descriptive
- consistent
- not overly verbose
- aligned with the visual identity you want to preserve

Good captions mention:
- subject identity or concept
- pose or framing
- lighting
- environment or background
- style if relevant

Bad captions:
- too generic
- contradictory
- keyword spam
- different naming for the same subject

### Dataset filtering approach

Recommended filtering pipeline:
1. Perceptual duplicate detection
2. Blur / sharpness filtering
3. Very low resolution filtering
4. Optional CLIP-based relevance filtering
5. Manual final review

### Example dataset loader contract

`training/dataset.py` should expose something like:

```python
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

@dataclass
class DatasetItem:
    image_path: Path
    caption: str

class CaptionedImageDataset(Dataset):
    def __init__(self, root_dir: str, captions_file: str | None = None):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int):
        ...

def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 0):
    ...
```

### Practical filtering implementation

A useful filtering heuristic is to compute:
- image size
- Laplacian variance for blur
- perceptual hash or SSIM-like duplicate detection
- optional CLIP similarity against caption or class prompt

## Training Pipeline

### Core approach

Use Hugging Face Diffusers with LoRA fine-tuning on the UNet.

Keep the training loop minimal:
- load dataset
- load base model
- attach LoRA adapters
- optimize only LoRA parameters
- save checkpoints every N steps or epochs
- never do validation image generation in the loop

### CPU optimization rules

On CPU-only systems:
- use batch size 1
- use gradient accumulation
- keep dataloader workers low or zero if I/O is unstable
- use cached latents if storage permits
- avoid frequent image decoding during training
- keep validation fully separate

### Suggested hyperparameters

Start here:
- base model: Stable Diffusion 1.5
- learning rate: `1e-5`
- epochs: `5` to `15`
- LoRA rank: `8` or `16`
- batch size: `1`
- gradient accumulation: `4` or `8`
- checkpoint interval: every `100` to `250` steps
- validation interval: offline only

### Mixed precision

On CPU, mixed precision usually does not help much and can be unsupported or unstable. Prefer standard float32 unless you have verified bfloat16 support and an actual performance gain.

### Caching latents

If the dataset is small and stable, cache image latents to disk or to a lightweight local cache.

This reduces repeated VAE encoding and helps CPU runs a lot.

Tradeoff:
- faster training
- more disk usage
- less flexibility if preprocessing changes

## Training Script Design

`train/train_lora.py` should:
- read a config file or CLI arguments
- load the filtered dataset
- instantiate the base model
- wrap LoRA adapters on attention layers
- train only adapter parameters
- save checkpoints
- write logs to disk
- exit cleanly with training metadata

### Suggested training config

```python
from dataclasses import dataclass

@dataclass
class TrainConfig:
    base_model: str = "runwayml/stable-diffusion-v1-5"
    dataset_dir: str = "data/filtered"
    caption_file: str | None = "data/captions/captions.jsonl"
    output_dir: str = "models/lora"
    learning_rate: float = 1e-5
    epochs: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    lora_rank: int = 8
    checkpoint_steps: int = 200
    seed: int = 42
```

### Training loop outline

```python
for epoch in range(config.epochs):
    for batch in dataloader:
        loss = training_step(batch)
        loss.backward()

        if step % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if global_step % config.checkpoint_steps == 0:
            save_checkpoint(...)
```

### Important rule

Do not call image generation from inside the training loop.

That is the main reason CPU training becomes unusable.

## Validation Strategy

Validation must be a separate script.

### Why separate validation matters

- training stays deterministic and simpler
- no long blocking calls during training
- you can run validation only when you need it
- validation can be scheduled on a different machine later
- easier to compare checkpoints consistently

### Validation script behavior

`val/validate.py` should:
- load a chosen checkpoint
- load a fixed set of prompts
- generate a small number of images per prompt
- save images to disk
- compute CLIP score
- write results to JSON or CSV

### Suggested validation prompts

Use a fixed evaluation prompt set:
- one prompt for subject identity
- one for style fidelity
- one for background consistency
- one for pose variation
- one for generalization

Keep the prompt list stable across runs so scores are comparable.

### Validation output contract

The script should produce:
- generated image files
- one JSON file with per-image CLIP scores
- one summary file with mean score and checkpoint metadata

## CLIP-Based Evaluation

Use CLIP as a lightweight proxy for prompt adherence.

### What it measures

CLIP similarity helps estimate:
- prompt-image alignment
- whether a checkpoint is improving
- which checkpoints are better than others

### What it does not measure well

CLIP alone does not capture:
- face quality
- identity consistency
- aesthetic quality
- fine-grained realism

So it should be used with checkpoint comparison and manual spot checks.

### Suggested scoring flow

For each checkpoint:
1. generate images from fixed prompts
2. compute CLIP similarity for each image-prompt pair
3. compute average score per checkpoint
4. track best checkpoint over time

### Example contract for `evaluation/clip_score.py`

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ClipResult:
    image_path: str
    prompt: str
    score: float


def score_image_text_pairs(image_paths, prompts, model_name="openai/clip-vit-base-patch32"):
    ...
```

## Best-Checkpoint Selection

Do not pick the best checkpoint only by training loss.

Better approach:
- use training loss as a stability signal
- use CLIP score as the first automatic quality metric
- inspect the top 2 or 3 checkpoints manually

Recommended rule:
- save the checkpoint with the best mean CLIP score
- also keep the latest checkpoint
- keep a small rolling window of checkpoints

## Logging and Tracking

Log these values:
- global step
- epoch
- loss
- learning rate
- checkpoint path
- validation CLIP mean score
- validation CLIP per prompt

Save logs as:
- JSONL for machine parsing
- CSV for quick inspection
- plain text summary for humans

## Performance Optimizations

### CPU bottleneck reduction

Use these tactics:
- resize images once during preprocessing
- cache filtered dataset metadata
- cache latents when possible
- keep workers low if disk is slow
- avoid repeatedly reloading the base model
- disable expensive validation in training

### Memory reduction

Use:
- small batch size
- gradient accumulation
- attention slicing if supported
- no unnecessary pipeline components during training
- no validation generation until offline evaluation

### Dataloader efficiency

For CPU systems, a practical choice is:
- `num_workers=0` or `1`
- `pin_memory=False`
- `persistent_workers=False`

If the dataset is small and local SSD is available, a single worker is often enough.

## Recommended APRIL Integration

Make the pipeline easy for APRIL to call with:
- one training entry point
- one validation entry point
- JSON config files
- structured output directories
- machine-readable result files

Suggested runtime artifacts:
- `models/lora/checkpoint-XXXX/`
- `models/lora/best/`
- `training/logs/train.jsonl`
- `training/logs/validation.jsonl`
- `evaluation/clip_scores.csv`

## Practical Execution Plan

### Phase 1
- clean dataset down to 50 to 120 high-quality images
- create consistent captions
- build dataset loader

### Phase 2
- implement CPU-friendly LoRA training
- save checkpoints only
- no validation generation inside training

### Phase 3
- implement offline validation
- generate fixed-prompt images from checkpoints
- compute CLIP scores

### Phase 4
- choose best checkpoint
- manually inspect top candidates
- integrate into APRIL

## Recommended Default Setup

If starting fresh, use this baseline:
- model: Stable Diffusion 1.5
- LoRA rank: 8
- learning rate: 1e-5
- epochs: 10
- batch size: 1
- gradient accumulation: 4
- checkpoint interval: 200 steps
- validation: separate script only
- metric: CLIP score + manual review

## Final Recommendation

For CPU-only training, the best balance is not to force validation into the training loop.

Instead:
- train cheaply and consistently
- save checkpoints often
- evaluate offline with CLIP
- keep only the best-performing checkpoints

That gives you a pipeline that is both practical and scalable for APRIL.
