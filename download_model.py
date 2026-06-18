"""
Download ONNX model files for deployment.
==========================================
Downloads the ONNX version of all-MiniLM-L6-v2 from Hugging Face Hub.
Run this during the build step (before the app starts).

Files downloaded:
  - onnx_model/model.onnx      (~30 MB)
  - onnx_model/tokenizer.json   (~700 KB)
"""

import os


def download_model(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                   output_dir: str = 'onnx_model'):
    """Download ONNX model and tokenizer from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    os.makedirs(output_dir, exist_ok=True)

    print(f"[download_model] Downloading ONNX model: {model_name}")

    # Download the ONNX model file
    # Try 'onnx/model.onnx' first (newer repos), then 'model.onnx'
    onnx_downloaded = False
    for onnx_filename in ['onnx/model.onnx', 'model.onnx']:
        try:
            path = hf_hub_download(
                repo_id=model_name,
                filename=onnx_filename,
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            # Move to standard location if nested
            final_path = os.path.join(output_dir, 'model.onnx')
            if path != final_path and os.path.exists(path):
                import shutil
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                shutil.move(path, final_path)
                # Clean up nested dir if created
                nested_dir = os.path.join(output_dir, 'onnx')
                if os.path.isdir(nested_dir):
                    shutil.rmtree(nested_dir)
            print(f"  ✓ ONNX model: {onnx_filename} → {final_path}")
            onnx_downloaded = True
            break
        except Exception as e:
            print(f"  ✗ {onnx_filename}: {e}")
            continue

    if not onnx_downloaded:
        # If no pre-exported ONNX exists, export from PyTorch
        print("  → No pre-exported ONNX found, converting from PyTorch...")
        _export_onnx(model_name, output_dir)

    # Download tokenizer
    for tok_file in ['tokenizer.json']:
        try:
            path = hf_hub_download(
                repo_id=model_name,
                filename=tok_file,
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            print(f"  ✓ Tokenizer: {tok_file}")
        except Exception as e:
            print(f"  ✗ {tok_file}: {e}")

    # Verify
    model_path = os.path.join(output_dir, 'model.onnx')
    tokenizer_path = os.path.join(output_dir, 'tokenizer.json')

    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"\n[download_model] ✓ Complete! Model: {model_size:.1f} MB in {output_dir}/")
    else:
        missing = []
        if not os.path.exists(model_path):
            missing.append('model.onnx')
        if not os.path.exists(tokenizer_path):
            missing.append('tokenizer.json')
        print(f"\n[download_model] ✗ Missing files: {', '.join(missing)}")
        raise FileNotFoundError(f"Failed to download: {', '.join(missing)}")


def _export_onnx(model_name: str, output_dir: str):
    """Fallback: export ONNX from PyTorch model (needs torch + transformers)."""
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
        model.save_pretrained(output_dir)
        print(f"  ✓ Exported ONNX model via optimum")
    except ImportError:
        print("  ✗ Cannot export: 'optimum' not installed.")
        print("    Install with: pip install optimum[onnxruntime]")
        raise


if __name__ == '__main__':
    download_model()
