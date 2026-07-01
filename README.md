# RunPod Flash - Speech to Text Demo

Transcribe Vietnamese audio using **faster-whisper (large-v3-turbo)** deployed on RunPod Flash serverless GPU, with a Streamlit frontend for demo and latency benchmarking.

## Architecture

```
Browser --> Streamlit (Community Cloud) --> RunPod Flash Endpoint (GPU)
                                            faster-whisper large-v3-turbo
```

## Project Structure

```
runpod_endpoint/
  handler.py          # RunPod Flash endpoint (deploy from WSL2)
app.py                # Streamlit app entry point
runpod_client.py      # RunPod API client with timing
benchmark.py          # Benchmark runner with statistics
requirements.txt      # Streamlit dependencies
packages.txt          # System dependencies (ffmpeg)
```

## Setup

### 1. Deploy RunPod Endpoint (WSL2)

```bash
pip install runpod-flash
flash login
cd runpod_endpoint
python handler.py <test_audio.mp3>   # deploys endpoint on first run
```

Note the **Endpoint ID** from the output.

### 2. Run Streamlit Locally

Create `.streamlit/secrets.toml`:

```toml
[runpod]
api_key = "rpa_..."
endpoint_id = "<your-endpoint-id>"
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo.
3. Set the secrets (same format as `secrets.toml`) in the app settings.
4. Share the URL with your mentor.

## Features

- Upload audio files (mp3/wav/m4a) or record from microphone
- Vietnamese and English transcription
- Per-request latency breakdown: total, queue, execution, network
- Benchmark mode: run N requests with aggregate statistics (mean, median, P90/P95/P99, cold start vs warm)
- Interactive charts showing latency distribution and breakdown
