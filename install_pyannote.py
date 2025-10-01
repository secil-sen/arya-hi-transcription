#!/usr/bin/env python3
"""
Script to install the correct version of pyannote.audio and check if it's working.
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{description}")
    print("=" * len(description))
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    print("Installing pyannote.audio for Arya transcription service")
    print("=" * 55)

    # Check current environment
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version}")

    # Install pyannote.audio
    success = run_command(
        "pip install \"pyannote.audio>=3.1.1\"",
        "Installing pyannote.audio"
    )

    if not success:
        print("WARNING: Installation may have failed. Trying alternative method...")
        success = run_command(
            "pip install --no-deps \"pyannote.audio>=3.1.1\"",
            "Installing pyannote.audio without dependencies"
        )

    # Test the installation
    print("\nTesting pyannote.audio installation...")
    test_code = '''
try:
    from pyannote.audio import Pipeline
    print("✓ Pipeline import successful")
    print(f"✓ Pipeline.from_pretrained available: {hasattr(Pipeline, 'from_pretrained')}")

    from pyannote.audio import Model
    print("✓ Model import successful")
    print(f"✓ Model.from_pretrained available: {hasattr(Model, 'from_pretrained')}")

    print("\\n✓ pyannote.audio installation appears to be working!")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\\nPlease run the following commands manually:")
    print("  pip install torch torchaudio")
    print("  pip install 'pyannote.audio>=3.1.1'")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
'''

    run_command(f'python -c "{test_code}"', "Testing pyannote.audio")

    print("\n" + "=" * 55)
    print("Installation complete!")
    print("\nNext steps:")
    print("1. Make sure your .env file has HUGGINGFACE_TOKEN set")
    print("2. Accept the license terms at:")
    print("   - https://hf.co/pyannote/embedding")
    print("   - https://hf.co/pyannote/speaker-diarization-3.1")
    print("3. Run your transcription service")

if __name__ == "__main__":
    main()