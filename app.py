"""Streamlit app for RunPod Flash Speech-to-Text demo."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from runpod_client import RunPodClient
from benchmark import run_benchmark

st.set_page_config(
    page_title="RunPod Flash STT Demo",
    page_icon=None,
    layout="wide",
)

SUPPORTED_FORMATS = ["mp3", "wav", "m4a", "webm", "ogg", "flac"]
LANGUAGES = {"Vietnamese": "vi", "English": "en", "Auto Detect": None}


def get_client() -> RunPodClient:
    """Initialize RunPod client from Streamlit secrets."""
    return RunPodClient(
        api_key=st.secrets["runpod"]["api_key"],
        endpoint_id=st.secrets["runpod"]["endpoint_id"],
    )


def render_header():
    """Render the app header."""
    st.title("RunPod Flash - Speech to Text Demo")
    st.caption(
        "Transcribe audio using faster-whisper (large-v3-turbo) "
        "deployed on RunPod Flash serverless GPU."
    )


def render_audio_input():
    """Render audio input section: file upload and microphone recording."""
    col_upload, col_mic = st.columns(2)

    with col_upload:
        st.subheader("Upload Audio File")
        uploaded = st.file_uploader(
            "Choose an audio file",
            type=SUPPORTED_FORMATS,
            key="audio_upload",
        )

    with col_mic:
        st.subheader("Record from Microphone")
        recorded = st.audio_input("Click to record", key="audio_record")

    # Determine which audio source to use
    audio_bytes = None
    source = None

    if uploaded is not None:
        audio_bytes = uploaded.read()
        source = uploaded.name
        uploaded.seek(0)
        st.audio(uploaded)
    elif recorded is not None:
        audio_bytes = recorded.read()
        source = "Microphone recording"
        recorded.seek(0)
        st.audio(recorded)

    return audio_bytes, source


def render_timing_metrics(timing: dict, output: dict):
    """Render latency metrics as Streamlit metric cards."""
    st.subheader("Latency Breakdown")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Latency", f"{timing['total_latency']:.3f}s")
    c2.metric("Queue Time", f"{timing['queue_time']:.3f}s")
    c3.metric("Execution Time", f"{timing['execution_time']:.3f}s")
    c4.metric("Network Overhead", f"{timing['network_overhead']:.3f}s")

    c5, c6, c7 = st.columns(3)
    c5.metric("Model Load", f"{output.get('model_load_time', 0):.3f}s")
    c6.metric("Inference Only", f"{output.get('inference_time', 0):.3f}s")

    audio_dur = output.get("audio_duration", 0)
    infer_time = output.get("inference_time", 0)
    if audio_dur > 0 and infer_time > 0:
        rtf = infer_time / audio_dur
        c7.metric("Realtime Factor", f"{rtf:.2f}x")
    else:
        c7.metric("Audio Duration", f"{audio_dur:.1f}s")


def render_transcription_result(result: dict):
    """Render transcription text and segment details."""
    output = result["output"]

    st.subheader("Transcription Result")
    st.text_area(
        "Transcribed Text",
        value=output.get("text", ""),
        height=120,
        disabled=True,
    )

    col_lang, col_prob, col_dur = st.columns(3)
    col_lang.metric("Language", output.get("language", "N/A"))
    col_prob.metric("Confidence", f"{output.get('language_probability', 0):.1%}")
    col_dur.metric("Audio Duration", f"{output.get('audio_duration', 0):.1f}s")

    render_timing_metrics(result["timing"], output)

    # Segment details in expander
    segments = output.get("segments", [])
    if segments:
        with st.expander(f"Segments ({len(segments)})"):
            for seg in segments:
                st.text(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")


def render_benchmark_stats(stats: dict):
    """Render benchmark aggregate statistics table."""
    st.subheader("Aggregate Statistics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Latency", f"{stats['total_latency']['mean']:.3f}s")
    col2.metric("Median (P50)", f"{stats['total_latency']['median']:.3f}s")
    col3.metric("P95 Latency", f"{stats['total_latency']['p95']:.3f}s")
    col4.metric("Throughput", f"{stats['throughput_rps']:.2f} req/s")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Cold Start", f"{stats['cold_start_latency']:.3f}s")
    col6.metric("Warm Avg", f"{stats['warm_avg_latency']:.3f}s")
    col7.metric("Min", f"{stats['total_latency']['min']:.3f}s")
    col8.metric("Max", f"{stats['total_latency']['max']:.3f}s")

    # Full stats table
    with st.expander("All Statistics"):
        stats_table = {
            "Metric": [
                "Mean", "Median", "Std Dev",
                "Min", "Max",
                "P90", "P95", "P99",
                "Cold Start", "Warm Avg",
                "Throughput (req/s)", "Realtime Factor",
                "Audio Duration (s)", "Total Requests",
            ],
            "Value": [
                f"{stats['total_latency']['mean']:.3f}s",
                f"{stats['total_latency']['median']:.3f}s",
                f"{stats['total_latency']['std']:.3f}s",
                f"{stats['total_latency']['min']:.3f}s",
                f"{stats['total_latency']['max']:.3f}s",
                f"{stats['total_latency']['p90']:.3f}s",
                f"{stats['total_latency']['p95']:.3f}s",
                f"{stats['total_latency']['p99']:.3f}s",
                f"{stats['cold_start_latency']:.3f}s",
                f"{stats['warm_avg_latency']:.3f}s",
                f"{stats['throughput_rps']:.3f}",
                f"{stats['realtime_factor_mean']:.3f}x",
                f"{stats['audio_duration']:.1f}",
                str(stats["n_requests"]),
            ],
        }
        st.table(pd.DataFrame(stats_table))


def render_benchmark_charts(per_request: list):
    """Render latency charts from per-request data."""
    df = pd.DataFrame(per_request)

    # Latency per request bar chart
    st.subheader("Latency Per Request")
    fig_bar = go.Figure()
    colors = ["#e74c3c" if i == 0 else "#3498db" for i in range(len(df))]
    fig_bar.add_trace(go.Bar(
        x=df["request_num"],
        y=df["total_latency"],
        marker_color=colors,
        name="Total Latency",
    ))
    fig_bar.update_layout(
        xaxis_title="Request #",
        yaxis_title="Latency (seconds)",
        showlegend=False,
        height=350,
        annotations=[dict(
            text="Red = cold start",
            xref="paper", yref="paper",
            x=0.99, y=0.99,
            showarrow=False,
            font=dict(size=11, color="#e74c3c"),
        )],
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Stacked breakdown chart
    st.subheader("Latency Breakdown Per Request")
    fig_stack = go.Figure()
    fig_stack.add_trace(go.Bar(
        x=df["request_num"], y=df["queue_time"],
        name="Queue Time", marker_color="#f39c12",
    ))
    fig_stack.add_trace(go.Bar(
        x=df["request_num"], y=df["model_load_time"],
        name="Model Load", marker_color="#e74c3c",
    ))
    fig_stack.add_trace(go.Bar(
        x=df["request_num"], y=df["inference_time"],
        name="Inference", marker_color="#2ecc71",
    ))
    fig_stack.add_trace(go.Bar(
        x=df["request_num"], y=df["network_overhead"],
        name="Network", marker_color="#3498db",
    ))
    fig_stack.update_layout(
        barmode="stack",
        xaxis_title="Request #",
        yaxis_title="Time (seconds)",
        height=350,
    )
    st.plotly_chart(fig_stack, use_container_width=True)


def tab_transcribe():
    """Single transcription tab."""
    audio_bytes, source = render_audio_input()

    col_lang, col_btn = st.columns([1, 3])
    with col_lang:
        lang_label = st.selectbox("Language", list(LANGUAGES.keys()))
    language = LANGUAGES[lang_label]

    if audio_bytes is None:
        st.info("Upload an audio file or record from your microphone to begin.")
        return

    st.caption(f"Source: {source} | Size: {len(audio_bytes) / 1024:.1f} KB")

    if st.button("Transcribe", type="primary", key="btn_transcribe"):
        client = get_client()
        with st.spinner("Transcribing..."):
            try:
                result = client.transcribe(audio_bytes, language=language)
                if result["status"] == "COMPLETED":
                    render_transcription_result(result)
                else:
                    st.error(f"Job failed with status: {result['status']}")
            except Exception as e:
                st.error(f"Request failed: {e}")


def tab_benchmark():
    """Benchmark tab: run N requests and display statistics."""
    st.subheader("Upload Audio for Benchmark")
    uploaded = st.file_uploader(
        "Choose an audio file",
        type=SUPPORTED_FORMATS,
        key="bench_upload",
    )

    col_n, col_lang = st.columns(2)
    with col_n:
        n_requests = st.slider("Number of requests", 3, 20, 10)
    with col_lang:
        lang_label = st.selectbox("Language", list(LANGUAGES.keys()), key="bench_lang")
    language = LANGUAGES[lang_label]

    if uploaded is None:
        st.info("Upload an audio file to run the benchmark.")
        return

    audio_bytes = uploaded.read()
    st.caption(f"File: {uploaded.name} | Size: {len(audio_bytes) / 1024:.1f} KB")

    if st.button("Run Benchmark", type="primary", key="btn_benchmark"):
        client = get_client()
        progress = st.progress(0, text="Starting benchmark...")

        def on_progress(current, total):
            progress.progress(
                current / total,
                text=f"Request {current}/{total} completed",
            )

        try:
            result = run_benchmark(
                client, audio_bytes, n_requests, language, on_progress
            )
            progress.empty()

            render_benchmark_stats(result["stats"])
            render_benchmark_charts(result["per_request"])

            # Raw data expander
            with st.expander("Raw Per-Request Data"):
                st.dataframe(
                    pd.DataFrame(result["per_request"]),
                    use_container_width=True,
                )
        except Exception as e:
            progress.empty()
            st.error(f"Benchmark failed: {e}")


def main():
    """App entry point."""
    render_header()
    tab1, tab2 = st.tabs(["Transcription", "Benchmark"])
    with tab1:
        tab_transcribe()
    with tab2:
        tab_benchmark()


if __name__ == "__main__":
    main()
