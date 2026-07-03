"""RunPod Flash endpoint for voice bot (STT + TTS) on a single GPU."""

import asyncio
import base64
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from runpod_flash import Endpoint, GpuType, NetworkVolume

load_dotenv()
VOLUME_ID = os.environ.get("RUNPOD_VOLUME_ID", "")
VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume/models")
volume = NetworkVolume(id=VOLUME_ID) if VOLUME_ID else None

# Cache models in VRAM to avoid reload on each request
_model_cache = {
    "stt": None,
    "tts": None,
    "tts_speaker_manager": None
}

def get_stt_model():
    """Load Whisper STT model, using Network Volume as cache directory."""
    from faster_whisper import WhisperModel
    if _model_cache["stt"] is None:
        t = time.time()
        _model_cache["stt"] = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="float16",
            download_root=VOLUME_PATH,
            num_workers=4
        )
        print(f"STT model loaded in {time.time() - t:.2f}s")
    return _model_cache["stt"]

def get_tts_model():
    """Load Coqui VITS model for Vietnamese TTS. Much lighter than XTTS-v2."""
    if _model_cache["tts"] is None:
        import subprocess, sys
        # Install TTS dynamically on Linux server to avoid Windows cross-compile errors
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "TTS==0.22.0"],
            check=True
        )
        t = time.time()
        os.environ["TTS_HOME"] = VOLUME_PATH
        from TTS.api import TTS
        # Vietnamese VITS model from Coqui (~300MB, no voice cloning needed)
        _model_cache["tts"] = TTS(model_name="tts_models/vi/vivos/vits", gpu=True)
        print(f"TTS model loaded in {time.time() - t:.2f}s")
    return _model_cache["tts"]

def process_stt_single(audio_b64: str, language: str = "vi") -> dict:
    """Transcribe a single audio file from base64."""
    audio_bytes = base64.b64decode(audio_b64)
    tmp_path = f"/tmp/{uuid.uuid4()}.wav"
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    model = get_stt_model()
    t = time.time()
    segments_gen, info = model.transcribe(tmp_path, language=language, beam_size=5)

    segments = []
    for seg in segments_gen:
        segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text})

    inference_time = time.time() - t
    os.remove(tmp_path)
    full_text = " ".join(s["text"] for s in segments).strip()

    return {
        "text": full_text,
        "segments": segments,
        "language": info.language,
        "audio_duration": round(info.duration, 2),
        "inference_time": round(inference_time, 3),
    }

@Endpoint(
    name="stt-whisper-demo",
    gpu=GpuType.NVIDIA_A100_SXM4_80GB,
    workers=(0, 1),
    idle_timeout=6,
    volume=volume,
    dependencies=["faster-whisper", "numpy", "huggingface_hub"],
    system_dependencies=["ffmpeg"],
)
def voice_engine(action: str, payload: dict):
    """Unified endpoint handling both STT (Whisper) and TTS (VITS) on a single GPU."""
    if action == "stt":
        language = payload.get("language", "vi")

        # Batch mode: send multiple audio files at once
        if "audio_base64_list" in payload:
            b64_list = payload["audio_base64_list"]
            t_batch = time.time()
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_stt_single, b64, language) for b64 in b64_list]
                results = [f.result() for f in futures]
            return {
                "batch_results": results,
                "total_batch_time": round(time.time() - t_batch, 3),
                "batch_size": len(b64_list)
            }

        # Single file mode
        if "audio_base64" in payload:
            return process_stt_single(payload["audio_base64"], language)

        return {"error": "Missing audio data in payload"}

    elif action == "tts":
        text = payload.get("text", "")
        if not text:
            return {"error": "Missing text in payload"}

        model = get_tts_model()
        out_path = f"/tmp/out_{uuid.uuid4()}.wav"

        t = time.time()
        # VITS does not require a speaker wav - simple and fast
        model.tts_to_file(text=text, file_path=out_path)
        inference_time = time.time() - t

        with open(out_path, "rb") as f:
            out_b64 = base64.b64encode(f.read()).decode("utf-8")
        os.remove(out_path)

        return {
            "audio_base64": out_b64,
            "inference_time": round(inference_time, 3)
        }

    return {"error": f"Unknown action: {action}"}

async def main():
    """Placeholder for local debugging."""
    print("This handler is meant to be run via RunPod Flash.")

if __name__ == "__main__":
    asyncio.run(main())
