
# HAR Report Analyzer

This project provides a **HAR (HTTP Archive) analyzer** built on **Model Context Protocol (MCP)** standards.
User Can Upload a HAR file as input and our MCP server will provide in depth analysis of HAR file with reports and deep analytical insights.

It consists of:
- 🚀 **MCP Backend Server** (`mcp_har_server.py`) – FastAPI + Celery  
- 🎨 **Streamlit UI** (`HAR_Analyser_UI.py`) – interactive frontend  
- 📑 **JSON/HTML reports** with analysis & Plotly charts  
- 🔍 **MCP Inspector integration** for testing/debugging  

---

## ✨ Features
- Upload & analyze single or multiple `.har` files  
- Async task handling with **Celery + Redis** for large files  
- Rich metrics:  
  - ✅ Top domains  
  - ✅ Status & method breakdown  
  - ✅ Largest & slowest requests  
  - ✅ Requests over time  
  - ✅ Performance & caching indicators  
  - ✅ API request pass/fail  
- Auto-saves JSON reports in `reports/`  
- Visual charts using **Plotly**  

---

## 🛠️ Setup

### 1. Clone this repo
### 2. Go to root directory using cd 
### 3. Install Necessary packages
 ### pip install -r requirements.txt
### 4. Start the MCP Server Backend using below command:

  ### uvicorn mcp_har_server:app --host 127.0.0.1 --port 8006 --reload
  ### Backend runs at → http://127.0.0.1:8006

## 📦 Endpoints

- **`/health`** → Health check of MCP server
- **`/upload_analyze`** → Upload a single HAR file
- **`/upload_analyze_multi`** → Upload multiple HAR files
- **`/task/{task_id}`** → Poll Celery async result
- **`/reports/{id}`** → Get report JSON
- **`/visual_report/{id}`** → Get Plotly visualization


### 5.  Run HAR_Analyser_UI.py this file using below command: 

 ### streamlit run -m HAR_Analyser_UI.py
 
### 6. Upload the HAR File and get insights and report.

### 7: Test MCP Server using MCP Inspector :

**Run this command to test MCP Server:**

**streamlit run MCP_Inspector.py**





