"""RunPod Flash endpoint for speech-to-text using faster-whisper."""

import asyncio
from runpod_flash import Endpoint, GpuType


@Endpoint(
    name="stt-whisper-demo",
    gpu=GpuType.NVIDIA_RTX_A5000,
    workers=(0, 1),
    idle_timeout=300,
    dependencies=["faster-whisper", "numpy"],
    system_dependencies=["ffmpeg"],
)
def transcribe(audio_base64: str, language: str = "vi"):
    """Decode base64 audio, run faster-whisper inference, return transcription."""
    import base64
    import os
    import time

    from faster_whisper import WhisperModel

    audio_bytes = base64.b64decode(audio_base64)
    tmp_path = "/tmp/audio_input"
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    t_load = time.time()
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    model_load_time = time.time() - t_load

    t_infer = time.time()
    segments_gen, info = model.transcribe(tmp_path, language=language, beam_size=5)

    segments = []
    for seg in segments_gen:
        segments.append(
            {"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text}
        )
    inference_time = time.time() - t_infer

    os.remove(tmp_path)
    full_text = " ".join(s["text"] for s in segments).strip()

    return {
        "text": full_text,
        "segments": segments,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "audio_duration": round(info.duration, 2),
        "model_load_time": round(model_load_time, 3),
        "inference_time": round(inference_time, 3),
    }


async def main():
    """Quick test: transcribe a sample audio file."""
    import base64
    import sys

    if len(sys.argv) < 2:
        print("Usage: python handler.py <audio_file_path>")
        return

    with open(sys.argv[1], "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    result = await transcribe(audio_b64, language="vi")
    print(f"Text: {result['text']}")
    print(f"Language: {result['language']} ({result['language_probability']})")
    print(f"Audio duration: {result['audio_duration']}s")
    print(f"Model load: {result['model_load_time']}s")
    print(f"Inference: {result['inference_time']}s")


if __name__ == "__main__":
    asyncio.run(main())
