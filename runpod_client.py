"""Client wrapper module for interacting with RunPod serverless endpoints using the Flash SDK."""

import base64
import time
import os
import asyncio
from typing import List
from runpod_flash import Endpoint

class RunPodClient:
    """Client wrapper handles structured communication with the RunPod Voice Bot Engine."""

    def __init__(self, api_key: str, endpoint_id: str):
        """Initializes the RunPod client configuration.

        Args:
            api_key (str): Authentication token for the RunPod API.
            endpoint_id (str): Target serverless endpoint identifier.
        """
        self.endpoint_id = endpoint_id
        # RunPod Flash SDK implicitly requires the API key via environment variables
        os.environ["RUNPOD_API_KEY"] = api_key

    async def _call_endpoint_async(self, payload: dict) -> dict:

        t_start = time.time()
        ep = Endpoint(id=self.endpoint_id)
        job = await ep.run(payload)
        await job.wait()

        total_latency = time.time() - t_start
        return {
            "status": job._data.get("status", "COMPLETED"),
            "output": job.output if isinstance(job.output, dict) else {},
            "error": job._data.get("error"),
            "timing": {
                "total_latency": round(total_latency, 3),
                "queue_time": 0,
                "execution_time": round(total_latency, 3),
                "network_overhead": 0,
            },
        }

    def _call_endpoint(self, payload: dict) -> dict:
        """Synchronous block executor wrapping async endpoints for Streamlit runtime interoperability.

        Args:
            payload (dict): Runtime data to pass over the network layer.

        Returns:
            dict: Evaluation results map returned by the core serverless handler.
        """
        return asyncio.run(self._call_endpoint_async(payload))

    def transcribe(self, audio_bytes: bytes, language: str = "vi") -> dict:
        """Submits a single raw audio segment for Speech-to-Text inference processing.

        Args:
            audio_bytes (bytes): Binary audio streams extracted via presentation layer forms.
            language (str, optional): Target language ISO code. Defaults to "vi".

        Returns:
            dict: Computed text translation mappings and timestamps array payload.
        """
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        payload = {
            "action": "stt",
            "payload": {
                "audio_base64": audio_b64,
                "language": language,
            }
        }
        return self._call_endpoint(payload)

    def transcribe_batch(self, audios_bytes: List[bytes], language: str = "vi") -> dict:
        """Dispatches multiple audio byte segments concurrently to leverage GPU-level batched execution.

        Args:
            audios_bytes (List[bytes]): List of raw audio binaries gathered from files layout.
            language (str, optional): Target language ISO code. Defaults to "vi".

        Returns:
            dict: Compiled aggregate batch results map representing processed instances.
        """
        audios_b64 = [base64.b64encode(ab).decode("utf-8") for ab in audios_bytes]
        payload = {
            "action": "stt",
            "payload": {
                "audio_base64_list": audios_b64,
                "language": language,
            }
        }
        return self._call_endpoint(payload)

    def synthesize_speech(self, text: str, language: str = "vi") -> dict:
        """Invokes Text-to-Speech (TTS) models to clone or synthesize voice waveforms from strings.

        Args:
            text (str): Input text paragraph to be verbalized by the backend cluster.
            language (str, optional): Speech language properties pattern. Defaults to "vi".

        Returns:
            dict: Output map housing base64-encoded output audio data stream.
        """
        payload = {
            "action": "tts",
            "payload": {
                "text": text,
                "language": language,
            }
        }
        return self._call_endpoint(payload)