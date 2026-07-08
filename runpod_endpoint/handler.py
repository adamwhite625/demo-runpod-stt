"""RunPod Flash serverless endpoint handler hosting unified STT (Whisper) and TTS (VITS) services."""

import asyncio
import base64
import os
import time
import uuid
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from runpod_flash import Endpoint, GpuType, NetworkVolume

load_dotenv()

VOLUME_ID = os.environ.get("RUNPOD_VOLUME_ID", "")
VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume/models")

print("===== DEBUG VOLUME =====")
print("VOLUME_ID:", VOLUME_ID)
print("VOLUME_PATH:", VOLUME_PATH)

if os.path.exists(VOLUME_PATH):
    print("VOLUME CONTENT:")
    print(os.listdir(VOLUME_PATH))
else:
    print("VOLUME PATH DOES NOT EXIST")

volume = NetworkVolume(id=VOLUME_ID) if VOLUME_ID else None

# In-memory VRAM model caching layer
_model_cache = {"stt": None, "tts": None}


def get_stt_model():
    """Lazy-loads and caches the Whisper STT model configuration.

    Returns:
        WhisperModel: Instantiated Faster-Whisper model object.
    """
    if _model_cache["stt"] is None:
        t = time.time()
        from faster_whisper import WhisperModel

        _model_cache["stt"] = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="float16",
            download_root=VOLUME_PATH,
            num_workers=4,
        )
        print(f"STT model loaded in {time.time() - t:.2f}s")
    return _model_cache["stt"]


def get_tts_model():
    """Loads and caches Coqui TTS VITS model."""

    if _model_cache["tts"] is None:
        t = time.time()

        os.environ["TTS_HOME"] = VOLUME_PATH

        from TTS.api import TTS

        _model_cache["tts"] = TTS(
            model_name="tts_models/vi/vivos/vits",
            gpu=True
        )

        print(f"TTS model loaded in {time.time() - t:.2f}s")

    return _model_cache["tts"]


def process_stt_single(audio_b64: str, language: str = "vi") -> dict:
    """Decodes base64 payload and executes a single Speech-to-Text translation task.

    Args:
        audio_b64 (str): Base64 encoded input audio string.
        language (str): Target language ISO identifier string. Defaults to "vi".

    Returns:
        dict: Transcription text content strings, timeline segments, and breakdown performance metrics.
    """
    audio_bytes = base64.b64decode(audio_b64)
    tmp_path = f"/tmp/{uuid.uuid4()}.wav"

    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    model = get_stt_model()
    t = time.time()
    segments_gen, info = model.transcribe(
        tmp_path, language=language, beam_size=5
    )

    segments = []
    for seg in segments_gen:
        segments.append(
            {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text,
            }
        )

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
    dependencies=["faster-whisper",
    "numpy==1.26.4",
    "huggingface_hub",
    "coqui-tts==0.27.3",],
    system_dependencies=["ffmpeg", "espeak-ng"],
)
def voice_engine(action: str, payload: dict) -> dict:
    """Unified infrastructure dispatch entry point mapping transaction actions to underlying models."""
    if action == "stt":
        language = payload.get("language", "vi")

        # Handle batch payload schema using structural concurrent thread-pools
        if "audio_base64_list" in payload:
            b64_list = payload["audio_base64_list"]
            t_batch = time.time()
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(process_stt_single, b64, language)
                    for b64 in b64_list
                ]
                results = [f.result() for f in futures]
            return {
                "batch_results": results,
                "total_batch_time": round(time.time() - t_batch, 3),
                "batch_size": len(b64_list),
            }

        # Single instance processing mode fallback pipeline
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
        model.tts_to_file(text=text, file_path=out_path)
        inference_time = time.time() - t

        with open(out_path, "rb") as f:
            out_b64 = base64.b64encode(f.read()).decode("utf-8")
        os.remove(out_path)

        return {
            "audio_base64": out_b64,
            "inference_time": round(inference_time, 3),
        }

    return {"error": f"Unknown action: {action}"}


async def main():
    """Local initialization block for terminal runtime visibility definitions."""
    print("This handler is meant to be run via RunPod Flash.")


if __name__ == "__main__":
    asyncio.run(main())