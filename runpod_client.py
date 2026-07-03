"""Client wrapper for calling RunPod serverless endpoint."""

import base64
import time
from typing import List, Optional

import requests

class RunPodClient:
    """Handles communication with RunPod Voice Bot Engine endpoint."""

    def __init__(self, api_key: str, endpoint_id: str):
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _call_endpoint(self, payload: dict, timeout: int = 120) -> dict:
        t_start = time.time()
        response = requests.post(
            f"{self.base_url}/runsync",
            json=payload,
            headers=self.headers,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        
        job_id = data.get("id")
        status = data.get("status")
        
        # Poll if job is taking longer than 90s (Cold Start)
        while status in ["IN_QUEUE", "IN_PROGRESS"]:
            time.sleep(5)
            status_resp = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=self.headers,
                timeout=30
            )
            status_resp.raise_for_status()
            data = status_resp.json()
            status = data.get("status")

        total_latency = time.time() - t_start

        delay_ms = data.get("delayTime", 0)
        exec_ms = data.get("executionTime", 0)
        queue_time = delay_ms / 1000 if delay_ms else 0
        execution_time = exec_ms / 1000 if exec_ms else 0
        network_overhead = max(0, total_latency - queue_time - execution_time)

        return {
            "status": data.get("status"),
            "output": data.get("output", {}),
            "error": data.get("error"),
            "timing": {
                "total_latency": round(total_latency, 3),
                "queue_time": round(queue_time, 3),
                "execution_time": round(execution_time, 3),
                "network_overhead": round(network_overhead, 3),
            },
        }

    def transcribe(self, audio_bytes: bytes, language: str = "vi") -> dict:
        """STT: Single file"""
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        payload = {
            "input": {
                "action": "stt",
                "payload": {
                    "audio_base64": audio_b64,
                    "language": language,
                }
            }
        }
        return self._call_endpoint(payload)

    def transcribe_batch(self, audios_bytes: List[bytes], language: str = "vi") -> dict:
        """STT: Batch multiple files on server"""
        audios_b64 = [base64.b64encode(ab).decode("utf-8") for ab in audios_bytes]
        payload = {
            "input": {
                "action": "stt",
                "payload": {
                    "audio_base64_list": audios_b64,
                    "language": language,
                }
            }
        }
        # Batch might take longer
        return self._call_endpoint(payload, timeout=300)

    def synthesize_speech(self, text: str, language: str = "vi") -> dict:
        """TTS: Text to speech using Coqui VITS (Vietnamese)."""
        payload = {
            "input": {
                "action": "tts",
                "payload": {
                    "text": text,
                    "language": language,
                }
            }
        }
        return self._call_endpoint(payload)

    def health_check(self) -> bool:
        """Check if the endpoint is reachable."""
        try:
            resp = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=10,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False
