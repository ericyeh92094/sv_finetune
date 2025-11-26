#!/usr/bin/env python3
"""Test officially exported ONNX models with sherpa-onnx."""

from sherpa_onnx.offline_recognizer import OfflineRecognizer
import numpy as np

def test_model(model_path, tokens_path, model_name):
    """Test a single model."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")
    
    try:
        # Load model
        recognizer = OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=2,
            language="auto",
            use_itn=True,
        )
        print(f"✅ Model loaded successfully!")
        
        # Create test audio (silence)
        sample_rate = 16000
        duration = 2.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        # Run inference
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        recognizer.decode_stream(stream)
        
        result = stream.result
        print(f"✅ Inference completed")
        print(f"   Text: '{result.text}'")
        print(f"   Tokens: {len(result.tokens)}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    print("="*70)
    print("SHERPA-ONNX OFFICIAL EXPORT VERIFICATION")
    print("="*70)
    
    models = [
        ("V1 FP32", "./onnx_v1_fixed/sensevoice_v1_finetuned.onnx", 
         "./onnx_v1_fixed/sensevoice_v1_finetuned-tokens.txt"),
        ("V1 INT8", "./onnx_v1_fixed/sensevoice_v1_finetuned.int8.onnx",
         "./onnx_v1_fixed/sensevoice_v1_finetuned-tokens.txt"),
        ("V2 FP32", "./onnx_v2_fixed/sensevoice_v2_finetuned.onnx",
         "./onnx_v2_fixed/sensevoice_v2_finetuned-tokens.txt"),
        ("V2 INT8", "./onnx_v2_fixed/sensevoice_v2_finetuned.int8.onnx",
         "./onnx_v2_fixed/sensevoice_v2_finetuned-tokens.txt"),
    ]
    
    results = {}
    for name, model_path, tokens_path in models:
        success = test_model(model_path, tokens_path, name)
        results[name] = success
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} models passed")
    
    if passed == total:
        print("\n🎉 ALL MODELS WORK WITH SHERPA-ONNX!")
    else:
        print(f"\n⚠️  {total - passed} model(s) failed")
    
    return passed == total


if __name__ == "__main__":
    exit(0 if main() else 1)
