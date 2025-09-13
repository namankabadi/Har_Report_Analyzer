import os
import json
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="MCP Inspector", layout="wide")
st.title("🕵️ MCP Inspector - Test MCP Server")

MCP_SERVER_URL = st.secrets.get("MCP_SERVER_URL", "http://127.0.0.1:8006")


# -----------------------------
# Upload HAR to MCP
# -----------------------------
st.header("Upload HAR to MCP Server")
har_file = st.file_uploader("Select HAR file", type=["har"])
if har_file:
    files = {"har_file": (har_file.name, har_file.getvalue(), "application/json")}
    r = requests.post(f"{MCP_SERVER_URL}/upload_analyze", files=files)
    result = r.json()
    st.json(result)

    if result.get("mode") == "async":
        task_id = result["task_id"]
        st.info(f"File processed asynchronously. Task ID: {task_id}")
        if st.button("Poll Task"):
            status = requests.get(f"{MCP_SERVER_URL}/task/{task_id}").json()
            st.json(status)
    else:
        st.success("File processed synchronously ✅")

        # ✅ Download report for uploaded file
        st.download_button(
            label="💾 Download Uploaded Report as JSON",
            data=json.dumps(result, indent=2),
            file_name=f"{har_file.name}_report.json",
            mime="application/json"
        )


# -----------------------------
# Fetch Specific Report
# -----------------------------
st.header("Fetch Report by ID")
report_id = st.text_input("Enter Report ID")
if report_id:
    r = requests.get(f"{MCP_SERVER_URL}/reports/{report_id}")
    if r.ok:
        report = r.json()
        st.subheader("Full Report JSON")
        st.json(report)

        # ✅ Download button
        st.download_button(
            label="💾 Download Report as JSON",
            data=json.dumps(report, indent=2),
            file_name=f"{report_id}.json",
            mime="application/json"
        )

        # ✅ Pretty render if available
        if "all_requests" in report:
            all_requests = report["all_requests"]

            successes = [req for req in all_requests if 200 <= req.get("status", 0) < 300]
            failures = [req for req in all_requests if req.get("status", 0) >= 400]
            others = [
                req for req in all_requests
                if req.get("status", 0) not in range(200, 300) and req.get("status", 0) < 400
            ]

            st.write("✅ Success")
            st.dataframe(successes)
            if successes:
                st.download_button(
                    "⬇️ Download Success as CSV",
                    data=pd.DataFrame(successes).to_csv(index=False).encode("utf-8"),
                    file_name=f"{report_id}_success.csv",
                    mime="text/csv"
                )

            st.write("❌ Failures")
            st.dataframe(failures)
            if failures:
                st.download_button(
                    "⬇️ Download Failures as CSV",
                    data=pd.DataFrame(failures).to_csv(index=False).encode("utf-8"),
                    file_name=f"{report_id}_failures.csv",
                    mime="text/csv"
                )

            st.write("ℹ️ Others")
            st.dataframe(others)
            if others:
                st.download_button(
                    "⬇️ Download Others as CSV",
                    data=pd.DataFrame(others).to_csv(index=False).encode("utf-8"),
                    file_name=f"{report_id}_others.csv",
                    mime="text/csv"
                )
    else:
        st.error(f"Error: {r.text}")


# -----------------------------
# Fetch All Reports
# -----------------------------
st.header("Fetch All Reports from MCP Server")
if st.button("Fetch All Reports"):
    r = requests.get(f"{MCP_SERVER_URL}/reports")
    if r.ok:
        report_ids = r.json()
        st.write(f"Found {len(report_ids)} reports")

        for rid in report_ids:
            st.subheader(f"Report ID: {rid}")
            r_detail = requests.get(f"{MCP_SERVER_URL}/reports/{rid}")
            if r_detail.ok:
                report = r_detail.json()
                st.json(report)

                # ✅ Download button for each report
                st.download_button(
                    label=f"💾 Download {rid}.json",
                    data=json.dumps(report, indent=2),
                    file_name=f"{rid}.json",
                    mime="application/json"
                )

                # ✅ Pretty render requests if available
                if "all_requests" in report:
                    all_requests = report["all_requests"]

                    successes = [req for req in all_requests if 200 <= req.get("status", 0) < 300]
                    failures = [req for req in all_requests if req.get("status", 0) >= 400]
                    others = [
                        req for req in all_requests
                        if req.get("status", 0) not in range(200, 300) and req.get("status", 0) < 400
                    ]

                    st.write("✅ Success")
                    st.dataframe(successes)
                    if successes:
                        st.download_button(
                            f"⬇️ Download {rid}_success.csv",
                            data=pd.DataFrame(successes).to_csv(index=False).encode("utf-8"),
                            file_name=f"{rid}_success.csv",
                            mime="text/csv"
                        )

                    st.write("❌ Failures")
                    st.dataframe(failures)
                    if failures:
                        st.download_button(
                            f"⬇️ Download {rid}_failures.csv",
                            data=pd.DataFrame(failures).to_csv(index=False).encode("utf-8"),
                            file_name=f"{rid}_failures.csv",
                            mime="text/csv"
                        )

                    st.write("ℹ️ Others")
                    st.dataframe(others)
                    if others:
                        st.download_button(
                            f"⬇️ Download {rid}_others.csv",
                            data=pd.DataFrame(others).to_csv(index=False).encode("utf-8"),
                            file_name=f"{rid}_others.csv",
                            mime="text/csv"
                        )
            else:
                st.error(f"Error fetching report {rid}: {r_detail.text}")
    else:
        st.error(f"Error: {r.text}")
