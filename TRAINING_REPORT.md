# APRIL Training Report

## Executive Summary

I ran a fast CPU-only LoRA experiment on Stable Diffusion 1.5 with both the UNet and the CLIP text encoder enabled. The pipeline is now working end to end: training completes, checkpoints are written, and validation produces CLIP scores and summary files.

The important result is that this specific high-rank, short-run configuration did not improve quality. The model trains without crashing, but the new checkpoint scores lower than the previous baseline. That means the implementation is correct, but the current hyperparameters are too aggressive or not yet tuned for this dataset.

## What Was Tested

- Base model: `runwayml/stable-diffusion-v1-5`
- Training mode: LoRA on UNet + CLIP text encoder
- Hardware target: CPU only
- Run length: `100` training steps
- Checkpointing: every `50` steps
- Validation: offline, separate from training
- Evaluation metric: CLIP image-text similarity

This was intentionally a fast run, meant to answer one question: does adding the text encoder help enough to justify the extra complexity and cost? At the moment, the answer is no for this configuration.

## Training Configuration

- Script: `training/train/train_lora.py`
- Steps: `100`
- LoRA rank / alpha: `16 / 32`
- Learning rate: `1e-5`
- Warmup: `20` steps
- Batch size: `1`
- Gradient accumulation: `4`
- Text encoder training: enabled
- Checkpoints saved at: `50` and `100`

Command used:

```powershell
c:/Users/aless/Desktop/codes/codice/IMAGE/.venv/Scripts/python.exe training/train/train_lora.py --max-train-steps 100 --checkpoint-steps 50 --log-every 10 --lora-rank 16 --lora-alpha 32 --train-text-encoder
```

## Dataset Situation

- Records loaded from manifest: `159`
- Cached features loaded into RAM: `183`
- Training ran from the already filtered/cached dataset pipeline

This matters because the dataset is no longer the bottleneck. The training loop is stable, and the cache is doing its job. The remaining problem is model quality, not pipeline failure.

## Training Behavior

Trainable parameter groups reported by the run:

- UNet trainable params: `256`
- Text encoder trainable params: `48`

Loss values logged during the run show an unstable pattern rather than a smooth steady improvement:

- step 10: `0.072454`
- step 20: `0.076384`
- step 30: `0.241367`
- step 40: `0.282571`
- step 50: `0.476960`
- step 60: `0.719111`
- step 70: `1.108075`
- step 80: `0.125654`
- step 90: `0.060259`
- step 100: `0.532710`

Interpretation:

- The run is learning something, but the signal is noisy.
- The higher rank and text-encoder fine-tuning likely made the optimization more sensitive.
- For a small CPU run, this can easily overshoot and then partially recover.

Final training metadata:

- Final step: `100`
- Final loss: `0.5327098965644836`
- Mean epoch loss: `0.305685`
- Text encoder trained: `true`

## Checkpoints Produced

- `training/output/personal-lora/checkpoint-50`
- `training/output/personal-lora/checkpoint-100`

The checkpoint structure is now split in a way that supports the two-adapter setup:

- UNet LoRA weights in `checkpoint-100/unet/`
- text encoder adapter in `checkpoint-100/text_encoder/`

## Validation Results

Validation script:

- `training/val/validate.py`

Validation artifacts created:

- `training/output/validation/clip_results.json`
- `training/output/validation/clip_results.csv`
- `training/output/validation/validation_summary.json`
- `training/output/validation/checkpoint-100.json`

Primary multi-prompt validation result:

- Checkpoint: `checkpoint-100`
- Mean CLIP score: `0.2229231297969818`

Quick single-prompt validation result:

- Prompt: `portrait of pstyle subject, cinematic lighting`
- CLIP score: `0.2075512409210205`

What this means:

- Validation works.
- The score is measurable and repeatable.
- The current experiment does not beat the previous baseline.

## Baseline Comparison

- Previous baseline CLIP score: `0.2528`
- New multi-prompt score: `0.2229`
- New single-prompt score: `0.2076`

Conclusion:

- The new text-encoder-enabled run is worse than the previous baseline on this metric.
- This does not mean the architecture is broken.
- It means the current hyperparameters are not the right tradeoff for this dataset.

## What Was Fixed

1. Training API mismatch
   - Fixed `pipeline.text_model` to `pipeline.text_encoder` in `training/train/train_lora.py`.

2. Text encoder checkpoint serialization
   - Replaced unsupported `save_lora_adapter(...)` with PEFT-compatible `save_pretrained(...)` for the text encoder.

3. Validation loading compatibility
   - Updated `training/val/validate.py` so the UNet and text encoder adapters are loaded with the current diffusers/PEFT behavior.

These fixes are important because they prove the new training path is operational, even though the score is not yet good.

## Current Situation

The situation is now clear:

- The training stack is working.
- The text encoder LoRA path is working.
- The validation stack is working.
- The latest fast run is not yet good enough in quality terms.

So the next step is not more debugging. It is tuning.

## Recommended Next Experiment

The strongest next move is to make the run less aggressive:

- lower LoRA rank to `8`
- lower alpha to `16`
- reduce learning rate to `5e-6`
- compare `checkpoint-50` and `checkpoint-100`
- keep the same prompts for validation so results stay comparable

If the goal is highest signal with the least wasted time, the next experiment should prioritize stability over capacity.

## Second Evaluation Pass

I reran validation in a more serious paired mode to check whether the low visual quality was just a validation artifact.

- Checkpoint tested: `checkpoint-200`
- Validation setup: `3` prompts, `3` images per prompt, same seeds for checkpoint and base model
- Compared against: the plain base model with no LoRA loaded

Results:

- Checkpoint-200 CLIP score: `0.2420651929246055`
- Base model CLIP score: `0.24193080597453648`
- Delta vs base: `+0.0001343869500690098`

Interpretation:

- The validation pipeline is not obviously broken.
- The checkpoint and the base model score almost the same under this metric.
- That matches the visual inspection: the model is not producing a clearly better result.
- The CLIP score here is too weak to explain the aesthetic quality difference on its own.

Practical conclusion:

- This is not a strong validation bug signal.
- It is more likely a model-quality issue, or a metric that is too coarse to reflect visual appeal.
