"""Benchmark runner for RunPod STT endpoint latency measurement."""

import numpy as np


def run_benchmark(client, audio_bytes: bytes, n_requests: int = 10,
                  language: str = "vi", progress_callback=None) -> dict:
    """Run N sequential transcription requests and collect timing metrics.

    Args:
        client: RunPodClient instance.
        audio_bytes: Raw audio file bytes.
        n_requests: Number of requests to run.
        language: Target language code.
        progress_callback: Optional callable(current, total) for progress updates.

    Returns:
        Dict with 'per_request' list and 'stats' aggregate statistics.
    """
    per_request = []

    for i in range(n_requests):
        result = client.transcribe(audio_bytes, language)
        output = result.get("output", {})

        per_request.append({
            "request_num": i + 1,
            "total_latency": result["timing"]["total_latency"],
            "queue_time": result["timing"]["queue_time"],
            "execution_time": result["timing"]["execution_time"],
            "network_overhead": result["timing"]["network_overhead"],
            "model_load_time": output.get("model_load_time", 0),
            "inference_time": output.get("inference_time", 0),
            "audio_duration": output.get("audio_duration", 0),
            "status": result["status"],
        })

        if progress_callback:
            progress_callback(i + 1, n_requests)

    stats = _compute_stats(per_request)
    return {"per_request": per_request, "stats": stats}


def _compute_stats(per_request: list) -> dict:
    """Compute aggregate statistics from per-request timing data."""
    latencies = np.array([r["total_latency"] for r in per_request])
    exec_times = np.array([r["execution_time"] for r in per_request])
    queue_times = np.array([r["queue_time"] for r in per_request])

    audio_dur = per_request[0].get("audio_duration", 0)
    realtime_factors = exec_times / audio_dur if audio_dur > 0 else exec_times

    return {
        "total_latency": {
            "mean": round(float(np.mean(latencies)), 3),
            "median": round(float(np.median(latencies)), 3),
            "min": round(float(np.min(latencies)), 3),
            "max": round(float(np.max(latencies)), 3),
            "std": round(float(np.std(latencies)), 3),
            "p90": round(float(np.percentile(latencies, 90)), 3),
            "p95": round(float(np.percentile(latencies, 95)), 3),
            "p99": round(float(np.percentile(latencies, 99)), 3),
        },
        "execution_time": {
            "mean": round(float(np.mean(exec_times)), 3),
            "median": round(float(np.median(exec_times)), 3),
        },
        "queue_time": {
            "mean": round(float(np.mean(queue_times)), 3),
            "max": round(float(np.max(queue_times)), 3),
        },
        "cold_start_latency": round(float(latencies[0]), 3),
        "warm_avg_latency": round(float(np.mean(latencies[1:])), 3)
        if len(latencies) > 1
        else round(float(latencies[0]), 3),
        "throughput_rps": round(float(len(latencies) / np.sum(latencies)), 3),
        "realtime_factor_mean": round(float(np.mean(realtime_factors)), 3),
        "audio_duration": audio_dur,
        "n_requests": len(per_request),
    }
