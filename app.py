"""Streamlit app for RunPod Flash Voice Bot Demo (Whisper STT + Piper TTS)."""

import streamlit as st 
import pandas as pd
import plotly.graph_objects as go
import base64

from runpod_client import RunPodClient
from benchmark import run_benchmark


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
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("RunPod Flash - Voice Bot AI Demo")
    st.caption(
        "Powered by Whisper (large-v3-turbo) and Piper TTS on a single serverless RTX 4090 GPU. "
        "Demonstrating GPU-level concurrent batching and multi-model hosting."
    )

def render_timing_metrics(timing: dict, output: dict):
    """Render latency metrics as Streamlit metric cards."""
    st.markdown("### ⏱Latency Breakdown")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Latency", f"{timing.get('total_latency', 0):.3f}s")
        c2.metric("Queue Time", f"{timing.get('queue_time', 0):.3f}s")
        c3.metric("Execution Time", f"{timing.get('execution_time', 0):.3f}s")
        c4.metric("Network Overhead", f"{timing.get('network_overhead', 0):.3f}s")

        st.divider()

        c5, c6, c7, c8 = st.columns(4)
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

    st.markdown("### Transcription Result")
    
    text = output.get("text", "")
    if text:
        st.info(text, icon="💬")
    else:
        st.warning("No speech detected.")

    with st.container(border=True):
        col_lang, col_prob, col_dur = st.columns(3)
        col_lang.metric("Language", output.get("language", "N/A").upper())
        col_prob.metric("Confidence", f"{output.get('language_probability', 0):.1%}")
        col_dur.metric("Audio Duration", f"{output.get('audio_duration', 0):.1f}s")

    st.write("") 
    if "timing" in result:
        render_timing_metrics(result["timing"], output)

    # Hiển thị Timeline Debug cho STT đơn lẻ
    if "debug" in output:
        st.write("")
        st.subheader("Backend Execution Timeline")
        if output["debug"].get("timeline"):
            df = pd.DataFrame(output["debug"]["timeline"])
            columns_to_show = [col for col in ["stage", "since_request", "since_boot"] if col in df.columns]
            st.dataframe(df[columns_to_show], use_container_width=True, hide_index=True)

    segments = output.get("segments", [])
    if segments:
        with st.expander(f"Segments ({len(segments)})"):
            for seg in segments:
                st.markdown(f"**[{seg.get('start', 0):.1f}s - {seg.get('end', 0):.1f}s]** {seg.get('text', '')}")

def tab_stt():
    """Batch Transcription Tab"""
    st.subheader("Speech-to-Text (GPU Batched Inference)")
    st.write("Tải lên một hoặc nhiều file âm thanh. Code backend sẽ gom lại và chạy song song đa luồng trực tiếp trên CUDA bằng CTranslate2.")
    
    uploaded_files = st.file_uploader(
        "Choose audio files",
        type=SUPPORTED_FORMATS,
        accept_multiple_files=True,
        key="audio_upload_batch",
    )
    
    col_lang, _ = st.columns([1, 3])
    with col_lang:
        lang_label = st.selectbox("Language", list(LANGUAGES.keys()))
    language = LANGUAGES[lang_label]

    if not uploaded_files:
        st.info("Upload files to begin.")
        return

    if st.button("Transcribe", type="primary"):
        client = get_client()
        with st.spinner(f"Transcribing {len(uploaded_files)} files concurrently..."):
            try:
                audios_bytes = [f.read() for f in uploaded_files]
                if len(audios_bytes) == 1:
                    result = client.transcribe(audios_bytes[0], language=language)
                    if result.get("status") == "COMPLETED":
                        if "error" in result["output"]:
                            st.error(result["output"]["error"])
                        else:
                            render_transcription_result(result)
                    else:
                        err = result.get("error")
                        st.error(f"Job failed with status: {result.get('status')}" + (f"\nDetails: {err}" if err else ""))
                else:
                    result = client.transcribe_batch(audios_bytes, language=language)
                    if result.get("status") == "COMPLETED":
                        out = result["output"]
                        if "error" in out:
                            st.error(out["error"])
                        else:
                            st.success(f"Batch {len(uploaded_files)} files completed in {out.get('total_batch_time', 0):.2f}s on GPU.")
                            st.json(result["timing"])
                            
                            # Hiển thị Timeline tổng của luồng Batch ở ngoài nếu có
                            if "debug" in out:
                                st.subheader("Batch Global Execution Timeline")
                                if out["debug"].get("timeline"):
                                    df_batch = pd.DataFrame(out["debug"]["timeline"])
                                    cols = [c for c in ["stage", "since_request", "since_boot"] if c in df_batch.columns]
                                    st.dataframe(df_batch[cols], use_container_width=True, hide_index=True)

                            for i, res in enumerate(out.get("batch_results", [])):
                                with st.expander(f"File: {uploaded_files[i].name} (Inference: {res.get('inference_time',0):.2f}s)"):
                                    single_res = {"output": res}
                                    render_transcription_result(single_res)
                    else:
                        err = result.get("error")
                        st.error(f"Batch Job failed with status: {result.get('status')}" + (f"\nDetails: {err}" if err else ""))
            except Exception as e:
                st.error(f"Request failed: {e}")

def tab_tts():
    """Text-to-Speech Tab using Piper VITS (Ultra lightweight & fast)."""
    st.subheader("Text-to-Speech (Piper VITS - Vietnamese)")
    st.write("Sử dụng cấu trúc Piper VITS tối ưu tốc độ kết hợp Network Volume để sinh giọng nói tiếng Việt tức thì.")

    text = st.text_area(
        "Văn bản cần đọc",
        "Xin chào! Hệ thống AI tổng hợp giọng nói Piper đã cấu hình thành công trên GPU RTX 4090.",
        height=120
    )

    if st.button("Synthesize Speech", type="primary"):
        if not text:
            st.warning("Vui lòng nhập văn bản.")
            return

        client = get_client()
        with st.spinner("Đang tổng hợp giọng nói qua Piper trên GPU..."):
            try:
                result = client.synthesize_speech(text, language="vi")
                if result.get("status") == "COMPLETED":
                    out = result["output"]
                    if "error" in out:
                        st.error(f"Model Error: {out['error']}")
                    else:
                        st.success(f"Synthesized in {out.get('inference_time', 0):.3f}s")
                        audio_b64 = out.get("audio_base64")
                        
                        # ========================================================
                        # 💻 ĐOẠN PHẦN THÊM VÀO ĐỂ CONSOLE LOG DEBUG TRÌNH DUYỆT (F12)
                        # ========================================================
                        import streamlit.components.v1 as components
                        
                        b64_len = len(audio_b64) if audio_b64 else 0
                        b64_preview = audio_b64[:100] if audio_b64 else "RỖNG"
                        
                        js_console_log = f"""
                        <script>
                            console.log("%c--- RUNPOD TTS FRONTEND DEBUG ---", "color: #3498db; font-weight: bold; font-size: 12px;");
                            console.log("Inference Time Backend:", "{out.get('inference_time', 0)}s");
                            console.log("Chuỗi Base64 Length:", {b64_len});
                            console.log("100 Ký tự đầu:", "{b64_preview}...");
                            if ({b64_len} < 100) {{
                                console.warn("CẢNH BÁO: Chuỗi Base64 có độ dài quá ngắn! Có thể dữ liệu nhị phân WAV bị rỗng.");
                            }}
                        </script>
                        """
                        components.html(js_console_log, height=0, width=0)
                        
                        print(f"\n[DEBUG app.py] Base64 Length nhận về: {b64_len} ký tự.")
                        print(f"[DEBUG app.py] Preview: {b64_preview}...\n")
                        # ========================================================

                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            st.audio(audio_bytes, format="audio/wav")
                        
                        with st.expander("Latency Details"):
                            st.json(result["timing"])
                            if "debug" in out:
                                st.write("")
                                st.subheader("Backend Execution Timeline")
                                if out["debug"].get("timeline"):
                                    df = pd.DataFrame(out["debug"]["timeline"])
                                    columns_to_show = [col for col in ["stage", "since_request", "since_boot"] if col in df.columns]
                                    st.dataframe(df[columns_to_show], use_container_width=True, hide_index=True)
                else:
                    err = result.get("error")
                    st.error(f"Job failed with status: {result.get('status')}" + (f"\nDetails: {err}" if err else ""))
            except Exception as e:
                st.error(f"Error: {e}")

def render_benchmark_stats(stats: dict):
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

def render_benchmark_charts(per_request: list):
    df = pd.DataFrame(per_request)
    st.subheader("Latency Per Request")
    fig_bar = go.Figure()
    colors = ["#e74c3c" if i == 0 else "#3498db" for i in range(len(df))]
    fig_bar.add_trace(go.Bar(x=df["request_num"], y=df["total_latency"], marker_color=colors))
    st.plotly_chart(fig_bar, width="stretch")

def tab_benchmark():
    st.subheader("Stress Test / Benchmark (STT Single Request Loop)")
    uploaded = st.file_uploader("Choose an audio file", type=SUPPORTED_FORMATS, key="bench_upload")
    
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
    if st.button("Run Benchmark", type="primary", key="btn_benchmark"):
        client = get_client()
        progress = st.progress(0, text="Starting benchmark...")
        def on_progress(current, total):
            progress.progress(current / total, text=f"Request {current}/{total} completed")
        try:
            result = run_benchmark(client, audio_bytes, n_requests, language, on_progress)
            progress.empty()
            render_benchmark_stats(result["stats"])
            render_benchmark_charts(result["per_request"])
        except Exception as e:
            progress.empty()
            st.error(f"Benchmark failed: {e}")

def main():
    """App entry point."""
    render_header()
    tab1, tab2, tab3 = st.tabs(["🎙️ STT Batch", "🔊 TTS VITS (Tiếng Việt)", "📈 Benchmark"])
    with tab1:
        tab_stt()
    with tab2:
        tab_tts()
    with tab3:
        tab_benchmark()

if __name__ == "__main__":
    main()