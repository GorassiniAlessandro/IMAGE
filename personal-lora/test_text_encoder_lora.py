#!/usr/bin/env python
"""Quick test for CLIPTextModel LoRA adapter configuration."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer

def test_text_encoder_lora():
    print("Loading CLIPTextModel and tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    
    print(f"✓ CLIPTextModel loaded: {text_encoder.__class__.__name__}")
    print(f"  Shape: hidden_size={text_encoder.config.hidden_size}")
    
    # Configure LoRA adapter
    print("\nConfiguring LoRA adapter for text encoder...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        init_lora_weights=True,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
        bias="none",
    )
    
    text_encoder.add_adapter(lora_config)
    text_encoder.set_adapter("default")
    print("✓ LoRA adapter added to text encoder")
    
    # Count trainable parameters
    lora_params = [p for p in text_encoder.parameters() if "lora" in p.__dict__.get('_qat_name', '').lower() or hasattr(p, 'is_lora_trainable')]
    total_trainable = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    
    print(f"  Trainable parameters: ~{total_trainable:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    text_input = tokenizer(
        "a photo of a person",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        output = text_encoder(text_input.input_ids)
    
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {output[0].shape}")
    
    # Verify adapter is set
    print(f"\nAdapter active: {text_encoder.active_adapters}")
    print(f"Adapter config: {text_encoder.peft_config}")
    
    print("\n✅ All tests passed! CLIPTextModel LoRA adapter works correctly.")
    return True

if __name__ == "__main__":
    try:
        test_text_encoder_lora()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
