"""Client wrapper for calling RunPod serverless endpoint."""

import base64
import time

import requests


class RunPodClient:
    """Handles communication with RunPod STT endpoint including timing."""

    def __init__(self, api_key: str, endpoint_id: str):
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def transcribe(self, audio_bytes: bytes, language: str = "vi") -> dict:
        """Send audio bytes to RunPod endpoint, return transcription with timing."""
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        payload = {
            "input": {
                "audio_base64": audio_b64,
                "language": language,
            }
        }

        t_start = time.time()
        response = requests.post(
            f"{self.base_url}/runsync",
            json=payload,
            headers=self.headers,
            timeout=120,
        )
        total_latency = time.time() - t_start

        response.raise_for_status()
        data = response.json()

        # RunPod returns delayTime and executionTime in milliseconds
        delay_ms = data.get("delayTime", 0)
        exec_ms = data.get("executionTime", 0)
        queue_time = delay_ms / 1000
        execution_time = exec_ms / 1000
        network_overhead = max(0, total_latency - queue_time - execution_time)

        return {
            "status": data.get("status"),
            "output": data.get("output", {}),
            "timing": {
                "total_latency": round(total_latency, 3),
                "queue_time": round(queue_time, 3),
                "execution_time": round(execution_time, 3),
                "network_overhead": round(network_overhead, 3),
            },
        }

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
