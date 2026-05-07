# Personal LoRA Report

## Scope
This report covers only the work moved into `personal-lora/`.
The folder is now the self-contained workspace for the LoRA experiment, with its own training code, validation code, dataset tooling, checkpoints, caches, and evaluation outputs.

## What Was Done
- Added CLIPTextModel LoRA support alongside the existing UNet LoRA path.
- Fixed adapter save/load so checkpoints can be restored correctly.
- Updated validation to handle checkpoints that do not include a text-encoder adapter.
- Added a CLIP-based smoke test for the text encoder LoRA flow.
- Reorganized the repository so the whole training stack lives under `personal-lora/training/`.
- Moved the experiment data and generated outputs into the `personal-lora/` workspace.
- Removed empty leftover root folders after the move.

## Experiment Flow
The workspace now contains the full path from dataset preparation to training and validation:
- dataset preparation and manifest/cache generation
- LoRA training for UNet and optional text encoder
- checkpoint saving and adapter export
- validation against generated prompts
- paired comparison against the base model

## Validation Summary
The later validation pass compared the checkpoint against the base model on the same prompts and samples.
The CLIP scores ended up very close, which showed that the metric was not strongly separating visual quality in this setup.
That is why the workspace now keeps the comparison workflow available for future runs.

## Current Structure
- `personal-lora/training/` contains the code and scripts.
- `personal-lora/checkpoints/`, `personal-lora/cache/`, `personal-lora/logs/`, `personal-lora/eval/`, `personal-lora/validation/`, and `personal-lora/smoke/` are the generated experiment areas.
- `personal-lora/test_text_encoder_lora.py` is the smoke test for the text encoder adapter path.

## Notes
The intent of this folder is to keep the experiment isolated without hiding source code.
Only large generated artifacts should remain ignored; the training and validation code stays trackable.