"""RunPod Flash serverless endpoint hosting Whisper STT + Piper TTS."""

import asyncio
import base64
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from runpod_flash import Endpoint, GpuType, NetworkVolume

import piper
import inspect

print("\n========== PIPER DEBUG ==========")
print("piper module :", piper.__file__)
print("piper version:", getattr(piper, "__version__", "unknown"))

from piper import PiperVoice

print("PiperVoice class :", PiperVoice)
print("PiperVoice module:", PiperVoice.__module__)
print("PiperVoice source:", inspect.getfile(PiperVoice))

print("\nMethods:")
for m in dir(PiperVoice):
    if not m.startswith("_"):
        print(" -", m)

print("=================================\n")

from importlib.metadata import version, PackageNotFoundError

try:
    print("piper-tts package version:", version("piper-tts"))
except PackageNotFoundError:
    print("piper-tts package not found")

load_dotenv()

# =========================
# Debug timeline
# =========================

BOOT_TIME = time.time()


def create_debug():
    return {
        "boot_time": BOOT_TIME,
        "request_start": None,
        "timeline": [],
    }


def debug_log(debug, stage, **kwargs):
    if debug is None:
        return
    now = time.time()

    if debug["request_start"] is None:
        elapsed_request = 0.0
    else:
        elapsed_request = now - debug["request_start"]

    debug["timeline"].append({
        "timestamp": round(now, 3),
        "since_boot": round(now - BOOT_TIME, 3),
        "since_request": round(elapsed_request, 3),
        "stage": stage,
        **kwargs
    })


# =========================
# Volume config
# =========================

VOLUME_ID = os.environ.get("RUNPOD_VOLUME_ID", "")
VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume/models")


print("===== DEBUG VOLUME =====")
print("VOLUME_ID:", VOLUME_ID)
print("VOLUME_PATH:", VOLUME_PATH)


if os.path.exists(VOLUME_PATH):
    for root, dirs, files in os.walk(VOLUME_PATH):
        print(root, files[:5])
else:
    print("VOLUME PATH DOES NOT EXIST")


volume = NetworkVolume(id=VOLUME_ID) if VOLUME_ID else None


# =========================
# Model cache
# =========================

_model_cache = {
    "stt": None,
    "tts": None
}


# =========================
# Whisper STT
# =========================

def get_stt_model(debug=None):
    if _model_cache["stt"] is None:
        if debug:
            debug_log(debug, "stt_model_loading")
            
        from faster_whisper import WhisperModel
        t = time.time()

        whisper_path = (
            "/runpod-volume/models/"
            "models--mobiuslabsgmbh--"
            "faster-whisper-large-v3-turbo/"
            "snapshots/"
            "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
        )

        print("Loading Whisper:", whisper_path)

        _model_cache["stt"] = WhisperModel(
            whisper_path,
            device="cuda",
            compute_type="float16",
            num_workers=4,
        )

        print(f"Whisper loaded: {time.time()-t:.2f}s")
        
        if debug:
            debug_log(
                debug,
                "stt_model_loaded",
                load_time=round(time.time() - t, 3),
            )
    elif debug:
        debug_log(debug, "stt_model_cache_hit")

    return _model_cache["stt"]



# =========================
# Piper TTS
# =========================

def get_tts_model(debug=None):
    if _model_cache["tts"] is None:
        if debug:
            debug_log(debug, "tts_model_loading")
            
        from piper import PiperVoice
        t = time.time()

        model_path = os.path.join(VOLUME_PATH, "piper", "vi_voice.onnx")
        
        print("Loading Piper:", model_path)

        # Thư viện Piper tự động tìm file .json nếu nó nằm chung thư mục và cùng tên
        _model_cache["tts"] = PiperVoice.load(model_path, use_cuda=True)
        print("\n========== MODEL DEBUG ==========")
        print("Model type:", type(_model_cache["tts"]))
        print("Model module:", type(_model_cache["tts"]).__module__)
        print("Model methods:")

        session = _model_cache["tts"].session

        print("Execution providers:", session.get_providers())

        try:
            print("Provider options:", session.get_provider_options())
        except Exception as e:
            print("Provider options unavailable:", e)

        for m in dir(_model_cache["tts"]):
            if not m.startswith("_"):
                print(" -"  , m)

        print("=================================\n")

        print(f"Piper loaded: {time.time()-t:.2f}s")
        
        if debug:
            debug_log(
                debug,
                "tts_model_loaded",
                load_time=round(time.time() - t, 3),
            )
    elif debug:
        debug_log(debug, "tts_model_cache_hit")

    return _model_cache["tts"]



# =========================
# STT processing
# =========================

def process_stt_single(audio_b64: str, language="vi", debug=None):
    audio_bytes = base64.b64decode(audio_b64)
    if debug:
        debug_log(debug, "audio_decoded")
        
    tmp_path = f"/tmp/{uuid.uuid4()}.wav"

    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    model = get_stt_model(debug)
    t = time.time()

    if debug:
        debug_log(debug, "whisper_inference_start")

    # Nhận dạng giọng nói
    segments_gen, info = model.transcribe(
        tmp_path,
        language=language,
        beam_size=5
    )

    segments = []
    for seg in segments_gen:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text,
        })

    inference_time = time.time() - t
    
    if debug:
        debug_log(
            debug,
            "whisper_inference_done",
            inference_time=round(inference_time, 3),
        )
        
    os.remove(tmp_path)

    return {
        "text": " ".join(x["text"] for x in segments).strip(),
        "segments": segments,
        "language": info.language,
        "audio_duration": round(info.duration, 2),
        "inference_time": round(inference_time, 3)
    }



# =========================
# RunPod Endpoint
# =========================

@Endpoint(
    name="voice-demo",
    gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
    workers=(0, 1),
    idle_timeout=6,
    volume=volume,
    dependencies=[
        "faster-whisper",
        "piper-tts",
        "numpy==1.26.4",
    ],
    system_dependencies=[
        "ffmpeg"
    ],
)
def voice_engine(action: str, payload: dict):
    debug = create_debug()
    debug["request_start"] = time.time()
    debug_log(
        debug,
        "handler_enter",
        action=action,
    )

    # =====================
    # Xử lý luồng STT
    # =====================
    if action == "stt":
        language = payload.get("language", "vi")

        # CASE 1: Xử lý Batch nhận diện nhiều file cùng lúc (Hỗ trợ app.py frontend)
        if "audio_base64_list" in payload:
            b64_list = payload["audio_base64_list"]
            t_batch = time.time()
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(process_stt_single, b64, language, debug)
                    for b64 in b64_list
                ]
                results = [f.result() for f in futures]
                
            return {
                "batch_results": results,
                "total_batch_time": round(time.time() - t_batch, 3),
                "batch_size": len(b64_list),
                "debug": debug,
            }

        # CASE 2: Xử lý đơn lẻ 1 file nhận diện
        if "audio_base64" in payload:
            result = process_stt_single(
                payload["audio_base64"],
                language,
                debug,
            )
            result["debug"] = debug
            return result

        return {"error": "Missing audio data in payload"}

    # =====================
    # Xử lý luồng TTS
    # =====================
    elif action == "tts":
        import io
        import wave

        text = payload.get("text", "")

        if not text:
            return {"error": "Missing text"}

        model = get_tts_model(debug)

        t = time.time()

        # Tạo WAV trong bộ nhớ
        wav_buffer = io.BytesIO()

        if debug:
            debug_log(debug, "tts_inference_start")

        with wave.open(wav_buffer, "wb") as wav_file:
            model.synthesize_wav(text, wav_file)

        wav_bytes = wav_buffer.getvalue()

        inference_time = time.time() - t

        if debug:
            debug_log(
                debug,
                "tts_inference_done",
                inference_time=round(inference_time, 3),
            )

        audio_base64 = base64.b64encode(wav_bytes).decode()

        return {
            "audio_base64": audio_base64,
            "inference_time": round(inference_time, 3),
            "debug": debug,
        }

    return {"error": f"Unknown action: {action}"}


async def main():
    print("RunPod Flash handler ready.")


if __name__ == "__main__":
    asyncio.run(main())