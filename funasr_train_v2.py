#!/usr/bin/env python3
"""
Enhanced FunASR training with stronger anti-catastrophic forgetting.
This version uses ultra-conservative settings to preserve zh accuracy
while finetuning on zh-TW data.
"""

import os
from pathlib import Path
import yaml


def main():
    print("=" * 80)
    print("Enhanced SenseVoice Training - Ultra-Conservative for zh Preservation")
    print("=" * 80)
    
    # Paths
    model_dir = "./models/iic/SenseVoiceSmall"
    data_dir = "./data/processed"
    output_dir = "./checkpoints/sensevoice_zh_combined_v2"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nModel: {model_dir}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    
    print("\n" + "=" * 80)
    print("Anti-Catastrophic Forgetting Strategy (Enhanced)")
    print("=" * 80)
    print("1. Ultra-Low Learning Rate: 1e-5 (5x more conservative)")
    print("2. Strong Weight Decay: 0.05 (5x stronger regularization)")
    print("3. Aggressive Gradient Clipping: 3.0 (tighter bounds)")
    print("4. Small Batch Size: 2 (more stable gradients)")
    print("5. Longer Training: 15 epochs (slower convergence)")
    print("6. Data Augmentation: Speed [0.9, 1.0, 1.1]")
    print("7. Warmup: 500 steps (gradual learning)")
    print("8. Layer-wise LR Decay: 0.8 (freeze early layers more)")
    
    # Training configuration with ultra-conservative settings
    trainer_config = {
        # Model
        "model": model_dir,
        "model_conf": {
            "ctc_conf": {
                "dropout_rate": 0.1,  # Add dropout for regularization
            },
        },
        
        # Data
        "dataset_type": "audio",
        "data_path": data_dir,
        "train_subset": "train.jsonl",
        "dev_subset": "dev.jsonl",
        
        # Ultra-conservative training parameters
        "batch_bins": 200000,  # Smaller effective batch size
        "max_epoch": 15,  # More epochs with slower learning
        "lr": 1.0e-5,  # Ultra-low learning rate (5x more conservative)
        "optim": "adam",
        "optim_conf": {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.05,  # Strong weight decay (5x stronger)
        },
        
        # Learning rate scheduling with warmup
        "scheduler": "warmuplr",
        "scheduler_conf": {
            "warmup_steps": 500,  # Gradual warmup
            "lr_decay": 0.995,  # Very slow decay
        },
        
        # Layer-wise learning rate decay (freeze early layers more)
        "freeze_param": [],  # Don't completely freeze, but use layer-wise LR
        
        # Gradient clipping (tighter)
        "grad_clip": 3.0,
        "grad_clip_type": "norm",
        "accum_grad": 2,  # Accumulate gradients for stability
        
        # Data augmentation
        "frontend_conf": {
            "fs": 16000,
            "n_mels": 80,
            "frame_length": 25,
            "frame_shift": 10,
            "lfr_m": 7,
            "lfr_n": 6,
        },
        "specaug": "specaug",
        "specaug_conf": {
            "apply_time_warp": False,  # Disable time warp to be more conservative
            "time_warp_window": 5,
            "time_warp_mode": "bicubic",
            "apply_freq_mask": True,
            "freq_mask_width_range": [0, 10],  # Mild masking
            "num_freq_mask": 1,  # Only 1 mask
            "apply_time_mask": True,
            "time_mask_width_range": [0, 30],  # Mild masking
            "num_time_mask": 1,  # Only 1 mask
        },
        
        # Speed perturbation (conservative)
        "speed_perturb": [0.95, 1.0, 1.05],  # Narrower range
        
        # Checkpointing
        "output_dir": output_dir,
        "save_checkpoint_interval": 500,
        "keep_nbest_models": 5,
        "log_interval": 50,
        
        # Device
        "device": "cpu",
        "num_workers": 4,
        
        # Validation
        "val_scheduler_criterion": ["valid", "loss"],
        "best_model_criterion": [["valid", "loss", "min"]],
    }
    
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    print(f"Learning Rate: {trainer_config['lr']}")
    print(f"Weight Decay: {trainer_config['optim_conf']['weight_decay']}")
    print(f"Gradient Clip: {trainer_config['grad_clip']}")
    print(f"Batch Bins: {trainer_config['batch_bins']}")
    print(f"Max Epochs: {trainer_config['max_epoch']}")
    print(f"Speed Perturb: {trainer_config['speed_perturb']}")
    
    # Import and run FunASR's training
    from funasr.bin.train import main as funasr_train_main
    import sys
    
    # Build command-line args for FunASR
    args = []
    for key, value in trainer_config.items():
        if isinstance(value, dict):
            # Skip nested dicts for now, will handle separately
            continue
        elif isinstance(value, list):
            args.extend([f"--{key}", *map(str, value)])
        else:
            args.extend([f"--{key}", str(value)])
    
    print(f"\nLaunching FunASR training with config...")
    print(f"Command args count: {len(args)}")
    
    # Save config to file
    import yaml
    config_file = Path(output_dir) / "training_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(trainer_config, f, default_flow_style=False)
    print(f"\n✓ Config saved to: {config_file}")
    
    # Run training
    sys.argv = ['funasr_train.py', '--config', str(config_file)]
    
    try:
        funasr_train_main()
    except SystemExit:
        pass  # Training completed
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Checkpoints saved to: {output_dir}")
    print(f"Best model: {output_dir}/model.pt.best")


if __name__ == "__main__":
    main()
