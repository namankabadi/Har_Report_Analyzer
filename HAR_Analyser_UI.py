# app.py
import os
import io
import json
import time
import uuid
import zipfile
import requests
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from datetime import datetime
from typing import List, Tuple, Dict, Any

# -----------------------------
# Config
# -----------------------------
MCP_SERVER_URL = st.secrets.get("MCP_SERVER_URL", "http://127.0.0.1:8006")
REPORTS_DIR = "har_ui_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Streamlit page config
st.set_page_config(page_title="HAR Report Analyzer", layout="wide")

# -----------------------------
# Helpers / API functions
# -----------------------------
def upload_har_file(file) -> dict:
    """Upload a single HAR file to MCP server and return JSON response."""
    try:
        files = {"har_file": (file.name, file.getvalue(), "application/json")}
        r = requests.post(f"{MCP_SERVER_URL}/upload_analyze", files=files, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"ok": False, "error": str(e)}

def upload_multiple_hars(files_list) -> dict:
    """Upload multiple HAR files to MCP server. Returns server JSON or {'results': []} on failure."""
    multipart = []
    for f in files_list:
        try:
            multipart.append(("har_files", (f.name, f.getvalue(), "application/json")))
        except Exception as e:
            st.error(f"Error reading file {getattr(f,'name','<unknown>')}: {e}")
    if not multipart:
        return {"results": []}
    try:
        r = requests.post(f"{MCP_SERVER_URL}/upload_analyze_multi", files=multipart, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Upload failed: {e}")
        return {"results": []}

def poll_task(task_id: str) -> dict:
    try:
        r = requests.get(f"{MCP_SERVER_URL}/task/{task_id}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"state": "ERROR", "error": str(e)}

def save_report_locally(report: dict) -> str:
    rid = report.get("id") or str(uuid.uuid4())
    path = os.path.join(REPORTS_DIR, f"{rid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path

# -----------------------------
# Chart generation & export helpers (with xN relabeling)
# -----------------------------
def generate_charts(report: dict) -> Tuple[List[Tuple[str, Any]], Dict[str, str]]:
    """
    Return (charts, label_map).
    charts: list of (name, plotly_fig)
    label_map: { 'x0' : 'real-long-url-or-domain', ... }
    """
    analysis = report.get("analysis", {}) or {}
    charts: List[Tuple[str, Any]] = []
    label_map: Dict[str, str] = {}

    def relabel(values):
        new_labels = []
        for v in values:
            v = "" if pd.isna(v) else str(v)
            matched_key = None
            for k, real in label_map.items():
                if real == v:
                    matched_key = k
                    break
            if matched_key is None:
                short = f"x{len(label_map)}"
                label_map[short] = v
                new_labels.append(short)
            else:
                new_labels.append(matched_key)
        return new_labels

    if analysis.get("top_domains"):
        df = pd.DataFrame(list(analysis["top_domains"].items()), columns=["Domain", "Requests"])
        df = df.sort_values("Requests", ascending=False).reset_index(drop=True)
        df["Domain_label"] = relabel(df["Domain"])
        fig = px.bar(df, x="Domain_label", y="Requests", title="Top Domains by Request Count")
        fig.update_layout(xaxis_title="Domain (xN)", xaxis_tickangle=-25, margin=dict(t=50, b=100))
        charts.append(("top_domains", fig))

    if analysis.get("status_breakdown"):
        df = pd.DataFrame(list(analysis["status_breakdown"].items()), columns=["Status", "Count"])
        fig = px.pie(df, names="Status", values="Count", title="HTTP Status Code Breakdown")
        charts.append(("status_breakdown", fig))

    if analysis.get("method_breakdown"):
        df = pd.DataFrame(list(analysis["method_breakdown"].items()), columns=["Method", "Count"])
        fig = px.pie(df, names="Method", values="Count", title="HTTP Method Breakdown")
        charts.append(("method_breakdown", fig))

    if analysis.get("slowest_requests"):
        try:
            df = pd.DataFrame(analysis["slowest_requests"])
            if not df.empty and "time_ms" in df.columns:
                df = df.sort_values("time_ms", ascending=False).reset_index(drop=True)
                df["short"] = relabel(df["url"])
                fig = px.bar(df, x="short", y="time_ms", hover_data=["url", "status"],
                             title="Top Slowest Requests (ms)")
                fig.update_layout(xaxis_title="Request (xN)", xaxis_tickangle=-30, margin=dict(t=50, b=120))
                charts.append(("slowest_requests", fig))
        except Exception:
            pass

    if analysis.get("largest_responses"):
        try:
            df = pd.DataFrame(analysis["largest_responses"])
            if not df.empty and "responseSize" in df.columns:
                df = df.sort_values("responseSize", ascending=False).reset_index(drop=True)
                df["short"] = relabel(df["url"])
                fig = px.bar(df, x="short", y="responseSize", hover_data=["url", "status"],
                             title="Top Largest Responses (bytes)")
                fig.update_layout(xaxis_title="Request (xN)", xaxis_tickangle=-30, margin=dict(t=50, b=120))
                charts.append(("largest_responses", fig))
        except Exception:
            pass

    if analysis.get("avg_timings"):
        try:
            df = pd.DataFrame(list(analysis["avg_timings"].items()), columns=["Phase", "Time_ms"])
            fig = px.bar(df, x="Phase", y="Time_ms", title="Average Timing Phases (ms)")
            fig.update_layout(xaxis_tickangle=-20, margin=dict(t=50, b=80))
            charts.append(("avg_timings", fig))
        except Exception:
            pass

    if analysis.get("requests_over_time"):
        try:
            ts = analysis["requests_over_time"]
            df = pd.DataFrame(list(ts.items()), columns=["Time", "Requests"])
            df["Time"] = pd.to_datetime(df["Time"])
            df = df.sort_values("Time")
            fig = px.line(df, x="Time", y="Requests", title="Requests Over Time")
            fig.update_layout(margin=dict(t=50, b=80))
            charts.append(("requests_over_time", fig))
        except Exception:
            pass

    return charts, label_map

def fig_to_png_bytes(fig) -> bytes:
    try:
        return pio.to_image(fig, format="png")
    except Exception:
        return b""

def export_charts_zip(charts: List[Tuple[str, Any]], report_id: str) -> str:
    zip_path = os.path.join(REPORTS_DIR, f"{report_id}_charts.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, fig in charts:
            png = fig_to_png_bytes(fig)
            if png:
                zf.writestr(f"{name}.png", png)
    return zip_path

# -----------------------------
# UI: header + sidebar
# -----------------------------
I_COLOR = "#FF6A00"

st.markdown(
    f"""
    <div style='display:flex;align-items:center;gap:12px'>
      <div style='background:{I_COLOR};padding:10px;border-radius:8px;color:white;font-weight:700'>HAR</div>
      <div>
        <h2 style='margin:0'>HAR Analyzer</h2>
        <div style='color:gray;margin-top:2px'>Dashboard for HAR inspection & reporting</div>
      </div>
    </div>
    <hr/>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Controls")
menu = ["Single HAR Upload", "Multiple HAR Upload", "View Saved Reports"]
choice = st.sidebar.selectbox("Menu", menu)

st.sidebar.markdown("---")
st.sidebar.markdown("Upload HAR files, inspect analysis, download charts & JSON.")

# -----------------------------
# Request Status Tables Helper
# -----------------------------
def render_request_tables(report: dict):
    requests_data = report.get("analysis", {}).get("all_requests", [])
    if not requests_data:
        return

    df = pd.DataFrame(requests_data)
    if "url" not in df.columns or "status" not in df.columns:
        st.info("No detailed request data available in this report.")
        return

    success_df = df[df["status"].astype(str).str.startswith("2")]
    fail_df = df[df["status"].astype(str).str.startswith(("4", "5"))]
    other_df = df[~df.index.isin(success_df.index.union(fail_df.index))]

    table_style = """
    <style>
    .good-table {
        border-collapse: collapse;
        width: 100%;
        margin-top: 10px;
    }
    .good-table th, .good-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
        font-size: 14px;
    }
    .good-table th {
        background-color: #FF6A00;
        color: white;
    }
    .success-row { background-color: #e8f5e9; }
    .fail-row { background-color: #ffebee; }
    .other-row { background-color: #e3f2fd; }
    </style>
    """

    def df_to_html(df, row_class):
        rows = ""
        for _, row in df.iterrows():
            rows += f"<tr class='{row_class}'><td>{row['url']}</td><td>{row['status']}</td></tr>"
        return rows

    st.markdown(table_style, unsafe_allow_html=True)

    if not success_df.empty:
        st.subheader("✅ Successful Requests")
        html = f"<table class='good-table'><tr><th>Request URL</th><th>Status</th></tr>{df_to_html(success_df, 'success-row')}</table>"
        st.markdown(html, unsafe_allow_html=True)

    if not fail_df.empty:
        st.subheader("❌ Failed Requests")
        html = f"<table class='good-table'><tr><th>Request URL</th><th>Status</th></tr>{df_to_html(fail_df, 'fail-row')}</table>"
        st.markdown(html, unsafe_allow_html=True)

    if not other_df.empty:
        st.subheader("ℹ️ Other Requests")
        html = f"<table class='good-table'><tr><th>Request URL</th><th>Status</th></tr>{df_to_html(other_df, 'other-row')}</table>"
        st.markdown(html, unsafe_allow_html=True)

# -----------------------------
# Single HAR Upload
# -----------------------------
if choice == "Single HAR Upload":
    st.header("Upload a Single HAR File")
    col1, col2 = st.columns([3, 1])
    with col1:
        har_file = st.file_uploader("Select HAR file", type=["har", "json"])
    with col2:
        allow_async = st.checkbox("Allow async (large files)", value=True)

    if har_file:
        st.info("Uploading...")
        result = upload_har_file(har_file)

        report = None
        if not result.get("ok", True) and result.get("error"):
            st.error(f"Upload error: {result.get('error')}")
        else:
            if result.get("mode") == "async":
                task_id = result.get("task_id")
                st.success(f"File queued for async processing. Task ID: {task_id}")
                if allow_async:
                    st.info("Polling task for result...")
                    status = poll_task(task_id)
                    while status.get("state") not in ["SUCCESS", "FAILURE", "ERROR"]:
                        time.sleep(1)
                        status = poll_task(task_id)
                        st.text(f"Task State: {status.get('state')}")
                    if status.get("state") == "SUCCESS":
                        report = status.get("result")
                    else:
                        st.error(f"Task failed: {status.get('error') or status.get('result')}")
                else:
                    st.warning("Async processing enabled — check task endpoint later.")
            else:
                report = result.get("report") or result

        if report:
            st.success("Analysis Complete ✅")
            metrics = report.get("analysis", {}) or {}
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Total Requests", metrics.get("total_requests", 0))
            r2.metric("Total Bytes", metrics.get("total_bytes", 0))
            r3.metric("Avg Latency (ms)", round(metrics.get("avg_time_ms", 0), 2))
            https_pct = metrics.get("https_ratio")
            r4.metric("HTTPS %", f"{round(https_pct*100,2)}%" if https_pct is not None else "N/A")

            # ✅ Render request tables here
            render_request_tables(report)

            path = save_report_locally(report)
            with open(path, "rb") as f:
                st.download_button("Download Report JSON", f.read(), file_name=os.path.basename(path), mime="application/json")

            charts, label_map = generate_charts(report)
            if charts:
                cols = st.columns(2)
                for i, (name, fig) in enumerate(charts):
                    with cols[i % 2]:
                        st.subheader(fig.layout.title.text or name)
                        st.plotly_chart(fig, use_container_width=True)
                        png = fig_to_png_bytes(fig)
                        if png:
                            st.download_button(f"Download {name}.png", data=png, file_name=f"{report.get('id','report')}_{name}.png", mime="image/png")
                zip_path = export_charts_zip(charts, report.get("id", str(uuid.uuid4())))
                if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
                    with open(zip_path, "rb") as zf:
                        st.download_button("Download All Charts (ZIP)", zf.read(), file_name=os.path.basename(zip_path), mime="application/zip")
                else:
                    st.info("PNG export not available (kaleido may be missing).")

                if label_map:
                    with st.expander("Legend for x-axis labels (xN → full label)"):
                        for short, real in label_map.items():
                            st.write(f"**{short}** → {real}")

            with st.expander("View raw JSON report"):
                st.json(report)

# -----------------------------
# Multiple HAR Upload
# -----------------------------
elif choice == "Multiple HAR Upload":
    st.header("Upload Multiple HAR Files")
    har_files = st.file_uploader("Select HAR files", type=["har", "json"], accept_multiple_files=True)
    if har_files:
        st.info("Uploading multiple HARs...")
        results = upload_multiple_hars(har_files)
        res_list = results.get("results") if isinstance(results, dict) and "results" in results else results
        if isinstance(res_list, dict) and res_list.get("id"):
            res_list = [res_list]
        if not isinstance(res_list, list):
            st.error("Unexpected response format from server.")
            res_list = []

        for res in res_list:
            if isinstance(res, dict) and res.get("error"):
                st.error(f"Upload result error: {res.get('error')}")
                continue
            report = res.get("report") if isinstance(res, dict) and res.get("report") else res
            if not report:
                st.warning("No report returned for an item.")
                continue

            st.subheader(f"Report: {report.get('meta', {}).get('source_file', report.get('id','report'))}")
            path = save_report_locally(report)
            with open(path, "rb") as f:
                st.download_button("Download Report JSON", f.read(), file_name=os.path.basename(path), mime="application/json")

            charts, label_map = generate_charts(report)
            if charts:
                cols = st.columns(2)
                for i, (name, fig) in enumerate(charts):
                    with cols[i % 2]:
                        st.subheader(fig.layout.title.text or name)
                        st.plotly_chart(fig, use_container_width=True)
                        png = fig_to_png_bytes(fig)
                        if png:
                            st.download_button(f"Download {name}.png", data=png, file_name=f"{report.get('id','report')}_{name}.png", mime="image/png")
                if label_map:
                    with st.expander("Legend for x-axis labels (xN → full label)"):
                        for short, real in label_map.items():
                            st.write(f"**{short}** → {real}")

# -----------------------------
# View Saved Reports
# -----------------------------
else:
    st.header("Saved Reports")
    files = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith(".json")], reverse=True)
    if not files:
        st.info("No saved reports found. Use Upload & Analyze to create reports.")
    else:
        chosen = st.selectbox("Choose saved report", options=files)
        if chosen:
            path = os.path.join(REPORTS_DIR, chosen)
            with open(path, "r", encoding="utf-8") as rf:
                report = json.load(rf)

            metrics = report.get("analysis", {}) or {}
            a, b, c, d = st.columns(4)
            a.metric("Total Requests", metrics.get("total_requests", 0))
            b.metric("Total Bytes", metrics.get("total_bytes", 0))
            c.metric("Avg Latency (ms)", round(metrics.get("avg_time_ms", 0), 2))
            https_pct = metrics.get("https_ratio")
            d.metric("HTTPS %", f"{round(https_pct*100,2)}%" if https_pct is not None else "N/A")

            # ✅ Render request tables here
            render_request_tables(report)

            charts, label_map = generate_charts(report)
            for name, fig in charts:
                st.subheader(fig.layout.title.text or name)
                st.plotly_chart(fig, use_container_width=True)
                png = fig_to_png_bytes(fig)
                if png:
                    st.download_button(f"Download {name}.png", data=png, file_name=f"{report.get('id','report')}_{name}.png", mime="image/png")

            if label_map:
                with st.expander("Legend for x-axis labels (xN → full label)"):
                    for short, real in label_map.items():
                        st.write(f"**{short}** → {real}")

            with st.expander("Raw JSON report"):
                st.json(report)

# Footer
st.markdown("---")
