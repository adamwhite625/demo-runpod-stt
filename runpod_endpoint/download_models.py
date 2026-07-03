import os

def main():
    # Thư mục ổ cứng mạng khi được mount trên RunPod thường là /runpod-volume
    volume_path = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume/models")
    os.makedirs(volume_path, exist_ok=True)
    
    print("======================================")
    print(f"Bắt đầu tải Models vào ổ đĩa mạng: {volume_path}")
    print("======================================")
    
    # 1. Tải Whisper
    print("\n[1/2] Đang tải Whisper large-v3-turbo...")
    from faster_whisper import WhisperModel
    # Chỉ cần gọi khởi tạo, faster-whisper sẽ tự download nếu chưa có
    WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", download_root=volume_path)
    print("-> Tải Whisper hoàn tất.")
    
    # 2. Tải XTTS-v2
    print("\n[2/2] Đang tải XTTS-v2 (Coqui TTS)...")
    # Thay đổi biến môi trường TTS_HOME để ép thư viện tải model về ổ cứng mạng thay vì ổ hệ thống
    os.environ["TTS_HOME"] = volume_path
    from TTS.api import TTS
    # gpu=False vì lúc tải chưa cần nạp vào VRAM
    TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    print("-> Tải XTTS-v2 hoàn tất.")
    
    print("\n======================================")
    print("ALL DONE! Tất cả Models đã sẵn sàng trên Network Volume.")
    print("======================================")

if __name__ == "__main__":
    main()
