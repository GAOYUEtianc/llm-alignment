#!/usr/bin/env python3
"""
Download Qwen2.5-Math-1.5B model from HuggingFace.
This script is designed to be run on RunPod or other remote servers after git clone.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
import argparse


def download_model(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    local_dir: str = "models/Qwen2.5-Math-1.5B",
    token: str = None
):
    """
    Download model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model identifier
        local_dir: Local directory to save the model
        token: HuggingFace token (optional, for gated models)
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_name} to {local_dir}...")
    print("This may take several minutes depending on your internet speed.")
    print(f"Model size: ~3GB")

    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token,
            resume_download=True,
        )
        print(f"\n✓ Model successfully downloaded to {local_dir}")

        # Verify the download
        config_file = local_dir / "config.json"
        if config_file.exists():
            print(f"✓ Verification successful: config.json found")
        else:
            print(f"⚠ Warning: config.json not found, download may be incomplete")

    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. If the model is gated, provide a HuggingFace token:")
        print("   python download_model.py --token YOUR_HF_TOKEN")
        print("3. Make sure you have enough disk space (~3GB)")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen2.5-Math-1.5B model from HuggingFace"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="models/Qwen2.5-Math-1.5B",
        help="Local directory to save the model"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (optional)"
    )

    args = parser.parse_args()

    # Use environment variable if token not provided
    token = args.token or os.environ.get("HF_TOKEN")

    download_model(
        model_name=args.model_name,
        local_dir=args.local_dir,
        token=token
    )


if __name__ == "__main__":
    main()
