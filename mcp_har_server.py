#!/usr/bin/env python3
"""
mcp_har_server.py

Enhanced MCP-style HAR analyzer with Celery background tasks.

Endpoints:
- GET  /health
- POST /upload_analyze     -> multipart upload; if HAR small → sync; if large → Celery async
- GET  /task/{task_id}     -> poll async result
- POST /upload_analyze_multi
- POST /mcp
- POST /visual_report
- GET  /reports/
- GET  /reports/{id}
"""

import os
import json
import uuid
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, List
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import plotly.express as px

from celery import Celery
from celery.result import AsyncResult

# --- Configuration ---
REPORTS_DIR = os.environ.get("MCP_REPORTS_DIR", os.path.join(os.getcwd(), "reports"))
os.makedirs(REPORTS_DIR, exist_ok=True)

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
BACKEND_URL = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

LARGE_FILE_THRESHOLD_MB = int(os.environ.get("HAR_LARGE_FILE_MB", "50"))  # switch to async if >50MB

# Setup Celery
celery_app = Celery("har_tasks", broker=BROKER_URL, backend=BACKEND_URL)

# FastAPI app
app = FastAPI(title="MCP HAR Analyzer + Celery")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Helpers
# -----------------------------
def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""


def parse_har_bytes(har_bytes: bytes) -> dict:
    text = har_bytes.decode("utf-8", errors="ignore")
    return json.loads(text)


def har_to_dataframe(har_json: dict) -> pd.DataFrame:
    entries = har_json.get("log", {}).get("entries", [])
    rows = []
    for e in entries:
        started = e.get("startedDateTime")
        try:
            started_dt = datetime.fromisoformat(started.replace("Z", "+00:00")) if started else None
        except Exception:
            started_dt = None
        req = e.get("request") or {}
        resp = e.get("response") or {}
        timings = e.get("timings") or {}
        url = req.get("url") or ""
        row = {
            "startedDateTime": started,
            "started_dt": started_dt,
            "time_ms": e.get("time"),
            "method": req.get("method"),
            "url": url,
            "domain": extract_domain(url),
            "status": resp.get("status"),
            "statusText": resp.get("statusText"),
            "mimeType": (resp.get("content") or {}).get("mimeType"),
            "responseSize": (resp.get("content") or {}).get("size"),
            "dns_ms": timings.get("dns"),
            "connect_ms": timings.get("connect"),
            "send_ms": timings.get("send"),
            "wait_ms": timings.get("wait"),
            "receive_ms": timings.get("receive"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["responseSize"] = pd.to_numeric(df.get("responseSize"), errors="coerce")
        df["time_ms"] = pd.to_numeric(df.get("time_ms"), errors="coerce")
        df["main_domain"] = df["domain"].fillna("").apply(lambda x: x.split(":")[0].lower())
    return df

def summarize_df(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"total_requests": 0, "total_bytes": 0}

    # Core stats
    total_requests = int(len(df))
    total_bytes = int(df["responseSize"].sum(skipna=True))
    avg_time_ms = float(df["time_ms"].mean(skipna=True))

    # Method, status, domain
    method_breakdown = df["method"].value_counts().to_dict()
    status_breakdown = df["status"].value_counts().to_dict()
    top_domains = df["main_domain"].value_counts().head(10).to_dict()

    # Slowest / largest
    slowest_requests = (
        df.sort_values("time_ms", ascending=False)
        .head(5)[["url", "time_ms", "status"]]
        .to_dict(orient="records")
    )
    largest_responses = (
        df.sort_values("responseSize", ascending=False)
        .head(5)[["url", "responseSize", "status"]]
        .to_dict(orient="records")
    )

    # Avg timings
    avg_timings = {
        "dns_ms": float(df["dns_ms"].mean(skipna=True)),
        "connect_ms": float(df["connect_ms"].mean(skipna=True)),
        "send_ms": float(df["send_ms"].mean(skipna=True)),
        "wait_ms": float(df["wait_ms"].mean(skipna=True)),
        "receive_ms": float(df["receive_ms"].mean(skipna=True)),
    }

    https_ratio = float((df["url"].str.startswith("https")).mean())

    # Requests over time
    time_series = {}
    if "started_dt" in df.columns and df["started_dt"].notna().any():
        ts = df.dropna(subset=["started_dt"])
        time_series = (
            ts.set_index("started_dt")
            .resample("1T")
            .size()
            .to_dict()
        )

    # -------------------------------
    # 🔹 Production-grade extras
    # -------------------------------

    # Content type breakdown
    content_types = df["mimeType"].dropna().value_counts().to_dict()

    # Error analysis
    errors = df[df["status"] >= 400]
    error_breakdown = errors["status"].value_counts().to_dict()

    # Redirects
    redirects = df[df["status"].between(300, 399)]
    redirect_chains = redirects[["url", "status"]].to_dict(orient="records")

    # Caching indicators
    cacheable = df[df["status"].isin([200, 304])]
    cache_hits = int((df["status"] == 304).sum())
    cache_misses = int((df["status"] == 200).sum())
    caching = {
        "cacheable": int(len(cacheable)),
        "hits_304": cache_hits,
        "misses_200": cache_misses,
        "hit_ratio": float(cache_hits / (cache_hits + cache_misses))
        if (cache_hits + cache_misses) > 0
        else 0.0,
    }

    # Compression check (rough: check mimeType + response size)
    compressed = int(df[df["mimeType"].str.contains("text|json|javascript|css", na=False)].shape[0])
    compression_usage = {
        "compressible_requests": compressed,
        "compression_ratio": float(compressed / total_requests) if total_requests else 0.0,
    }
    # ✅ Full request list (URL + status)
    all_requests = df[["url", "status"]].fillna("").to_dict(orient="records")

    # Approx user-centric perf (rough estimates)
    avg_ttfb = float(df["wait_ms"].mean(skipna=True))
    est_page_load = float(df["time_ms"].sum(skipna=True)) / total_requests
    perf_kpis = {"avg_ttfb_ms": avg_ttfb, "est_page_load_ms": est_page_load}

    # Third-party requests
    first_domain = df["main_domain"].mode().iloc[0] if not df.empty else ""
    third_party = df[df["main_domain"] != first_domain]["main_domain"].value_counts().to_dict()

    # HTTP API requests (JSON / XHR / fetch)
    api_requests = df[df["mimeType"].str.contains("json|api|javascript", na=False, case=False)]
    api_pass = int(api_requests[api_requests["status"].between(200, 299)].shape[0])
    api_fail = int(api_requests[api_requests["status"] >= 400].shape[0])
    api_report = {
        "total_api_requests": int(len(api_requests)),
        "passed": api_pass,
        "failed": api_fail,
        "failures": api_requests[api_requests["status"] >= 400][["url", "status"]].to_dict(orient="records"),
    }

    return {
        "total_requests": total_requests,
        "total_bytes": total_bytes,
        "avg_time_ms": avg_time_ms,
        "top_domains": top_domains,
        "status_breakdown": status_breakdown,
        "method_breakdown": method_breakdown,
        "slowest_requests": slowest_requests,
        "largest_responses": largest_responses,
        "avg_timings": avg_timings,
        "https_ratio": https_ratio,
        "requests_over_time": clean_for_json(time_series),
        # 🔹 New sections
        "content_types": content_types,
        "all_requests": all_requests,
        "errors": error_breakdown,
        "redirects": redirect_chains,
        "caching": caching,
        "compression": compression_usage,
        "performance": perf_kpis,
        "third_party_requests": third_party,
        "api_requests": api_report,
    }


def clean_for_json(obj: Any) -> Any:
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# -----------------------------
# Celery Task
# -----------------------------
@celery_app.task(bind=True)
def process_har_task(self, har_bytes: bytes, source_file: str) -> Dict[str, Any]:
    har_json = parse_har_bytes(har_bytes)
    df = har_to_dataframe(har_json)
    summary = summarize_df(df)
    report = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "meta": {"source_file": source_file, "n_entries": int(len(df))},
        "analysis": clean_for_json(summary),
    }
    # Save to disk
    path = os.path.join(REPORTS_DIR, f"{report['id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/upload_analyze")
async def upload_analyze(har_file: UploadFile = File(...)):
    """
    If HAR < threshold → process inline.
    If HAR > threshold → push to Celery queue and return task_id.
    """
    content = await har_file.read()
    size_mb = len(content) / (1024 * 1024)

    if size_mb < LARGE_FILE_THRESHOLD_MB:
        har_json = parse_har_bytes(content)
        df = har_to_dataframe(har_json)
        summary = summarize_df(df)
        report = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "meta": {"source_file": har_file.filename, "n_entries": int(len(df))},
            "analysis": clean_for_json(summary),
        }
        return {"ok": True, "mode": "sync", "report": report}
    else:
        task = process_har_task.delay(content, har_file.filename)
        return {"ok": True, "mode": "async", "task_id": task.id, "file": har_file.filename, "size_mb": size_mb}
def clean_for_json(obj: Any) -> Any:
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}  # 🔹 force string keys
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

@app.post("/upload_analyze_multi")
async def upload_analyze_multi(har_files: List[UploadFile] = File(...)):
    """
    Accept multiple HAR files and return a list of results.
    Each file is analyzed either sync (if small) or async (if large).
    """
    results = []
    for har_file in har_files:
        try:
            content = await har_file.read()
            size_mb = len(content) / (1024 * 1024)

            if size_mb < LARGE_FILE_THRESHOLD_MB:
                # Process inline
                har_json = parse_har_bytes(content)
                df = har_to_dataframe(har_json)
                summary = summarize_df(df)
                report = {
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "meta": {"source_file": har_file.filename, "n_entries": int(len(df))},
                    "analysis": clean_for_json(summary),
                }
                # Save to disk
                path = os.path.join(REPORTS_DIR, f"{report['id']}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)

                results.append({"ok": True, "mode": "sync", "report": report})
            else:
                # Dispatch async task
                task = process_har_task.delay(content, har_file.filename)
                results.append({
                    "ok": True,
                    "mode": "async",
                    "task_id": task.id,
                    "file": har_file.filename,
                    "size_mb": size_mb,
                })
        except Exception as e:
            results.append({"ok": False, "error": str(e), "file": har_file.filename})

    return {"results": results}

@app.get("/task/{task_id}")
def get_task_result(task_id: str):
    """
    Poll Celery task status.
    """
    res: AsyncResult = AsyncResult(task_id, app=celery_app)
    if res.state == "PENDING":
        return {"state": res.state}
    elif res.state == "SUCCESS":
        return {"state": res.state, "result": res.result}
    elif res.state == "FAILURE":
        return {"state": res.state, "error": str(res.result)}
    else:
        return {"state": res.state}

@app.get("/reports/{report_id}")
def get_report(report_id: str):
    """Download previously saved report JSON"""
    path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    with open(path, "r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))



@app.get("/visual_report/{report_id}")
def visual_report(report_id: str):
    """Return Plotly HTML visualization of report"""
    path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    df = pd.DataFrame([report["analysis"]["top_domains"]]).T.reset_index()
    df.columns = ["Domain", "Requests"]

    fig = px.bar(df, x="Domain", y="Requests", title="Top Domains by Request Count")
    html = fig.to_html(full_html=False)

    return HTMLResponse(content=html)


@app.get("/health")
def health():
    return {"status": "ok", "service": "mcp_har_server", "reports_dir": REPORTS_DIR}
from fastapi import FastAPI
import os, json

REPORTS_DIR = "har_ui_reports"



@app.get("/reports")
def list_reports():
    """List all saved report IDs"""
    if not os.path.exists(REPORTS_DIR):
        return []
    files = [f.replace(".json", "") for f in os.listdir(REPORTS_DIR) if f.endswith(".json")]
    return files


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("mcp_har_server:app", host="127.0.0.1", port=8006, reload=True)
