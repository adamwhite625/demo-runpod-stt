"""Model pre-download script for caching heavy AI weights onto the RunPod Network Volume."""

import os

def main():
    """Main execution block to orchestrate pre-downloading and caching of ML models."""
    # Network volumes attached to RunPod serverless instances are typically mounted under /runpod-volume
    volume_path = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume/models")
    os.makedirs(volume_path, exist_ok=True)
    
    print("======================================")
    print(f"Starting model provisioning to Network Volume: {volume_path}")
    print("======================================")
    
    # 1. Fetching and Caching Whisper STT Model Artifacts
    print("\n[1/2] Fetching Whisper large-v3-turbo artifacts...")
    from faster_whisper import WhisperModel
    
    # Instantiating the class prompts faster-whisper to pull remote weights into download_root if absent
    WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", download_root=volume_path)
    print("-> Whisper model ingestion completed successfully.")
    
    # 2. Fetching and Caching XTTS-v2 Core Model Artifacts
    print("\n[2/2] Fetching XTTS-v2 (Coqui TTS) multilingual weights...")
    
    # Override TTS_HOME environment variable to force checkpoint storage onto the network storage partition
    os.environ["TTS_HOME"] = volume_path
    from TTS.api import TTS
    
    # Set gpu=False during initial download container provisioning phase to avoid early VRAM footprint lock allocation
    TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    print("-> XTTS-v2 model ingestion completed successfully.")
    
    print("\n======================================")
    print("ALL DONE! Target model definitions are cached and ready on the Network Volume.")
    print("======================================")

if __name__ == "__main__":
    main()