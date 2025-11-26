# SenseVoice zh-TW Finetuning Project

This project finetunes the FunASR SenseVoiceSmall model for Taiwanese Mandarin (zh-TW) using the Common Voice dataset.

## 📊 Dataset

- **Source**: Mozilla Common Voice v23.0 zh-TW
- **Total Audio**: 16.89 hours (1,013 minutes)
- **Samples**: 17,556 total
  - Training: 7,356 samples (6.73 hours)
  - Validation: 5,100 samples (4.85 hours)
  - Test: 5,100 samples (5.30 hours)
- **Average Duration**: 3.46 seconds per sample

## 🎯 Models Trained

### V1: zh-TW Focused Model
- **Location**: `checkpoints/sensevoice_zh_tw_funasr/`
- **Training**: 10 epochs
- **Strategy**: Conservative finetuning (lr=5e-5)
- **Samples Processed**: 73,560 (70.75 hours total exposure)
- **Goal**: Optimize for zh-TW performance

### V2: zh Preservation Model
- **Location**: `checkpoints/sensevoice_zh_combined_v2/`
- **Training**: 15 epochs
- **Strategy**: Ultra-conservative (lr=1e-5, wd=0.05)
- **Samples Processed**: 110,340 (106.13 hours total exposure)
- **Goal**: Improve zh-TW while preserving original zh accuracy

## 📦 ONNX Exports

### V1 ONNX Models
**Location**: `onnx_models_v1/`
- `sensevoice_v1_finetuned.onnx` (897 MB) - FP32 model
- `sensevoice_v1_finetuned.int8.onnx` (230 MB) - INT8 quantized
- `sensevoice_v1_finetuned-tokens.txt` - Vocabulary file

### V2 ONNX Models
**Location**: `onnx_models_v2/`
- `sensevoice_v2_finetuned.onnx` (897 MB) - FP32 model
- `sensevoice_v2_finetuned.int8.onnx` (230 MB) - INT8 quantized
- `sensevoice_v2_finetuned-tokens.txt` - Vocabulary file

**Features**:
- ✅ Compatible with sherpa-onnx
- ✅ IR version 8 (sherpa-onnx compatible)
- ✅ INT8 quantization (75% size reduction)
- ✅ Fixed sequence_mask type conflicts
- ✅ Full metadata included

## 🚀 Usage

### 1. Training (if needed)

```bash
# Activate environment
source venv/bin/activate

# Train V2 model (recommended)
funasr-train --config-path . --config-name config_v2
```

### 2. Export to ONNX

```bash
# Export V1 model
python export_to_onnx.py \
  --checkpoint ./checkpoints/sensevoice_zh_tw_funasr \
  --output-dir ./onnx_models_v1 \
  --model-name sensevoice_v1_finetuned

# Export V2 model
python export_to_onnx.py \
  --checkpoint ./checkpoints/sensevoice_zh_combined_v2 \
  --output-dir ./onnx_models_v2 \
  --model-name sensevoice_v2_finetuned
```

### 3. Test ONNX Models

```bash
# Test all 4 models (V1/V2, FP32/INT8)
python test_onnx.py
```

### 4. Inference with sherpa-onnx

```python
from sherpa_onnx.offline_recognizer import OfflineRecognizer
import numpy as np

# Load model
recognizer = OfflineRecognizer.from_sense_voice(
    model='./onnx_models_v2/sensevoice_v2_finetuned.int8.onnx',
    tokens='./onnx_models_v2/sensevoice_v2_finetuned-tokens.txt',
    num_threads=4,
    language="zh",  # or "auto"
    use_itn=True,
)

# Load audio (16kHz, float32)
audio = np.load('your_audio.npy')  # or use librosa/soundfile

# Run inference
stream = recognizer.create_stream()
stream.accept_waveform(16000, audio)
recognizer.decode_stream(stream)

# Get result
result = stream.result
print(f"Text: {result.text}")
```

## 🛠️ Key Scripts

- **`config_v2.yaml`** - V2 training configuration (ultra-conservative)
- **`funasr_train_v2.py`** - V2 training script (if needed)
- **`export_to_onnx.py`** - Export finetuned models to ONNX
- **`test_onnx.py`** - Verify ONNX models work with sherpa-onnx

## 🔧 Technical Details

### Training Strategy
- **V1**: Standard finetuning with lr=5e-5
- **V2**: Ultra-conservative with lr=1e-5, wd=0.05, grad_clip=3.0
- **Batch size**: 2 (memory constraints)
- **Optimizer**: AdamW
- **Scheduler**: Warmup linear

### ONNX Export Key Fixes
1. **sequence_mask patching**: Fixed type conflicts (tensor(float) vs tensor(int64))
2. **IR version downgrade**: Changed from IR v10 to v8 for sherpa-onnx
3. **Opset 13**: Used for better compatibility
4. **INT8 quantization**: Dynamic quantization on MatMul operations

## 📂 Directory Structure

```
.
├── checkpoints/              # Trained PyTorch models
│   ├── sensevoice_zh_tw_funasr/      # V1 model
│   └── sensevoice_zh_combined_v2/    # V2 model
├── data/                     # Dataset
│   ├── processed/            # Processed JSONL files
│   └── cv-corpus-23.0-*/     # Common Voice raw data
├── onnx_models_v1/          # V1 ONNX exports
├── onnx_models_v2/          # V2 ONNX exports
├── backup_old_files/        # Archived intermediate files
├── config_v2.yaml           # Training configuration
├── export_to_onnx.py        # ONNX export script
└── test_onnx.py             # ONNX verification script
```

## ✅ Verification Results

All 4 ONNX models successfully verified:
- ✅ V1 FP32 - Loads and runs in sherpa-onnx
- ✅ V1 INT8 - Loads and runs in sherpa-onnx
- ✅ V2 FP32 - Loads and runs in sherpa-onnx
- ✅ V2 INT8 - Loads and runs in sherpa-onnx

## 🎯 Recommendations

**For Production Use**:
- **Use V2 INT8 model** (`onnx_models_v2/sensevoice_v2_finetuned.int8.onnx`)
- Benefits: 75% smaller size, preserves original zh performance, good zh-TW accuracy
- Framework: sherpa-onnx (designed specifically for these models)

**For Best zh-TW Performance**:
- Use V1 models if zh-TW accuracy is the only priority
- Trade-off: May impact original zh language performance

## 📝 Notes

- All old/intermediate files backed up in `backup_old_files/`
- Models require sherpa-onnx (not generic ONNXRuntime)
- FP32 models are ~897MB, INT8 models are ~230MB
- Training checkpoints are PyTorch format (~893MB each)

## 🔗 References

- [FunASR](https://github.com/modelscope/FunASR)
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [Common Voice](https://commonvoice.mozilla.org/)
