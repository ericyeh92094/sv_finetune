#!/usr/bin/env python3
"""
Export finetuned SenseVoice model using k2-fsa/sherpa-onnx official export method.
Based on: https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/sense-voice/export-onnx.py
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from funasr import AutoModel


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place."""
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


def modified_forward(
    self,
    x: torch.Tensor,
    x_length: torch.Tensor,
    language: torch.Tensor,
    text_norm: torch.Tensor,
):
    """Modified forward pass for ONNX export."""
    language_query = self.embed(language).unsqueeze(1)
    text_norm_query = self.embed(text_norm).unsqueeze(1)

    event_emo_query = self.embed(torch.LongTensor([[1, 2]])).repeat(x.size(0), 1, 1)

    x = torch.cat((language_query, event_emo_query, text_norm_query, x), dim=1)
    x_length = x_length + 4

    encoder_out, encoder_out_lens = self.encoder(x, x_length)
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]

    ctc_logits = self.ctc.ctc_lo(encoder_out)

    return ctc_logits


def sequence_mask_fixed(lengths, maxlen=None, dtype=torch.float32, device=None):
    """Fixed sequence_mask that avoids type conflicts in ONNX export."""
    if maxlen is None:
        maxlen = lengths.max()
    
    # Cast lengths to int64 to ensure consistent types
    lengths = lengths.long()
    row_vector = torch.arange(0, int(maxlen), 1, dtype=torch.int64).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    
    # Both are now int64, so Less operation will have consistent types
    mask = row_vector < matrix
    mask = mask.detach()

    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


def load_cmvn(filename) -> Tuple[str, str]:
    """Load CMVN statistics from file."""
    neg_mean = None
    inv_stddev = None

    with open(filename) as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]

            if neg_mean is None:
                neg_mean = ",".join(t)
            else:
                inv_stddev = ",".join(t)

    return neg_mean, inv_stddev


def generate_tokens(params, output_file="tokens.txt"):
    """Generate tokens file from tokenizer."""
    sp = params["tokenizer"].sp
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(sp.vocab_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")
    
    print(f"Generated {output_file} with {sp.vocab_size()} tokens")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Export finetuned SenseVoice to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to finetuned checkpoint directory (e.g., ./checkpoints/sensevoice_zh_combined_v2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./onnx_export",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="model",
        help="Name prefix for output files",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Loading base SenseVoiceSmall model...")
    print("=" * 80)
    
    # Load base model using FunASR
    base_model = AutoModel(model="iic/SenseVoiceSmall", device="cpu")
    model = base_model.model
    model.eval()
    
    # Get params from base model
    params = {
        "tokenizer": base_model.kwargs.get('tokenizer'),
        "frontend_conf": base_model.kwargs.get('frontend_conf', {}),
        "config": None,
    }
    
    # Load finetuned checkpoint
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_file = checkpoint_dir / "model.pt"
    if not checkpoint_file.exists():
        checkpoint_file = checkpoint_dir / "model.pt.best"
    
    if checkpoint_file.exists():
        print(f"\nLoading finetuned weights from: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        
        # Extract model state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        print("✓ Loaded finetuned weights")
    else:
        print(f"\n⚠️  Warning: No checkpoint found at {checkpoint_dir}")
        print("Using base model weights")
    
    print("\n" + "=" * 80)
    print("Generating tokens.txt...")
    print("=" * 80)
    
    tokens_file = output_dir / f"{args.model_name}-tokens.txt"
    generate_tokens(params, str(tokens_file))
    
    print("\n" + "=" * 80)
    print("Exporting to ONNX...")
    print("=" * 80)
    
    # Patch sequence_mask function to fix type conflicts
    import funasr.models.sense_voice.model as sense_voice_module
    original_sequence_mask = sense_voice_module.sequence_mask
    sense_voice_module.sequence_mask = sequence_mask_fixed
    print("✓ Patched sequence_mask to fix ONNX type conflicts")
    
    # Replace forward method for ONNX export
    model.__class__.forward = modified_forward
    
    # Create dummy inputs
    x = torch.randn(2, 100, 560, dtype=torch.float32)
    x_length = torch.tensor([80, 100], dtype=torch.int32)
    language = torch.tensor([0, 3], dtype=torch.int32)
    text_norm = torch.tensor([14, 15], dtype=torch.int32)
    
    opset_version = 13
    filename = output_dir / f"{args.model_name}.onnx"
    
    torch.onnx.export(
        model,
        (x, x_length, language, text_norm),
        str(filename),
        opset_version=opset_version,
        input_names=["x", "x_length", "language", "text_norm"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_length": {0: "N"},
            "language": {0: "N"},
            "text_norm": {0: "N"},
            "logits": {0: "N", 1: "T"},
        },
    )
    
    print(f"✓ Exported to {filename}")
    
    # Downgrade IR version for sherpa-onnx compatibility
    onnx_model = onnx.load(str(filename))
    if onnx_model.ir_version > 9:
        print(f"  Downgrading IR version from {onnx_model.ir_version} to 8 for sherpa-onnx compatibility")
        onnx_model.ir_version = 8
        onnx.save(onnx_model, str(filename))
        print(f"  ✓ IR version downgraded to 8")
    
    # Add metadata
    lfr_window_size = params["frontend_conf"].get("lfr_m", 7)
    lfr_window_shift = params["frontend_conf"].get("lfr_n", 6)
    
    cmvn_file = params["frontend_conf"].get("cmvn_file")
    if cmvn_file:
        neg_mean, inv_stddev = load_cmvn(cmvn_file)
    else:
        neg_mean = inv_stddev = ""
    
    vocab_size = params["tokenizer"].sp.vocab_size()
    
    # Get dictionaries from model
    lid_dict = getattr(model, 'lid_dict', {
        "auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13
    })
    textnorm_dict = getattr(model, 'textnorm_dict', {
        "withitn": 14, "woitn": 15
    })
    
    meta_data = {
        "lfr_window_size": lfr_window_size,
        "lfr_window_shift": lfr_window_shift,
        "normalize_samples": 0,
        "neg_mean": neg_mean,
        "inv_stddev": inv_stddev,
        "model_type": "sense_voice_ctc",
        "version": "2",  # QUInt8
        "model_author": "iic",
        "maintainer": "finetuned",
        "vocab_size": vocab_size,
        "comment": f"Finetuned SenseVoiceSmall from {args.checkpoint}",
        "lang_auto": lid_dict.get("auto", 0),
        "lang_zh": lid_dict.get("zh", 3),
        "lang_en": lid_dict.get("en", 4),
        "lang_yue": lid_dict.get("yue", 7),
        "lang_ja": lid_dict.get("ja", 11),
        "lang_ko": lid_dict.get("ko", 12),
        "lang_nospeech": lid_dict.get("nospeech", 13),
        "with_itn": textnorm_dict.get("withitn", 14),
        "without_itn": textnorm_dict.get("woitn", 15),
        "url": "https://github.com/FunAudioLLM/SenseVoice",
    }
    
    add_meta_data(filename=str(filename), meta_data=meta_data)
    print(f"✓ Added metadata to {filename}")
    
    print("\n" + "=" * 80)
    print("Quantizing to INT8...")
    print("=" * 80)
    
    filename_int8 = output_dir / f"{args.model_name}.int8.onnx"
    quantize_dynamic(
        model_input=str(filename),
        model_output=str(filename_int8),
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QUInt8,
    )
    
    print(f"✓ Quantized model saved to {filename_int8}")
    
    # Downgrade IR version for INT8 model too
    int8_model = onnx.load(str(filename_int8))
    if int8_model.ir_version > 9:
        print(f"  Downgrading INT8 model IR version from {int8_model.ir_version} to 8")
        int8_model.ir_version = 8
        onnx.save(int8_model, str(filename_int8))
        print(f"  ✓ INT8 model IR version downgraded to 8")
    
    # Restore original sequence_mask
    sense_voice_module.sequence_mask = original_sequence_mask
    
    print("\n" + "=" * 80)
    print("Export Summary")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"  - {args.model_name}.onnx ({os.path.getsize(filename) / 1024 / 1024:.1f} MB)")
    print(f"  - {args.model_name}.int8.onnx ({os.path.getsize(filename_int8) / 1024 / 1024:.1f} MB)")
    print(f"  - {args.model_name}-tokens.txt ({vocab_size} tokens)")
    print("\nYou can now use these files with sherpa-onnx for inference!")
    print("=" * 80)


if __name__ == "__main__":
    torch.manual_seed(20241126)
    main()
