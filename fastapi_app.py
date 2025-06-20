from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
import uvicorn
import json, re

"""fastapi_app.py – GST DOST backend micro‑service

Key endpoints
--------------
1. **POST /upload**      – store & index files.  Accepts optional `username` so we can tag
   ownership in `metadata_store` even though `streamlit_app.add_file_to_index` was built
   for Streamlit sessions.
2. **GET /files**        – list all indexed files (optionally filter by `username`).
3. **POST /query**       – natural‑language search across full documents (no page range UI).
4. **POST /email-severity** – classify an email + attachments into Low / Normal / …

The helper functions come directly from `streamlit_app.py` (same folder).  The only
incompatibility is FastAPI’s `UploadFile` exposing `.filename` instead of `.name`, which
we solve via the `StreamlitFileAdapter` wrapper.
"""

# ---------------------------------------------------------------------------
# Imports & helper wrappers
# ---------------------------------------------------------------------------

from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional

# Helper logic from Streamlit backend
from app2 import (
    add_file_to_index,
    load_index_and_metadata,
    query_documents_with_page_range,
    metadata_store,
    save_index_and_metadata,  # <-- ensure persistence after FastAPI uploads
    call_llm_api,
    call_claude_api,
    call_gpt_api,
    call_novalite_api,
    call_deepseek_api,
)

# ---------------------------------------------------------------------------
# Adapter: make FastAPI UploadFile look like Streamlit UploadedFile
# ---------------------------------------------------------------------------

class StreamlitFileAdapter:
    """Wrap `fastapi.UploadFile` so downstream helpers see `.name` & `.read`."""

    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename  # attribute expected by Streamlit helpers
        self.filename = uf.filename  # keep original attr

    def read(self, *args, **kwargs):
        return self._uf.file.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._uf.file.seek(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._uf, item)

# ---------------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------------

app = FastAPI(title="GST DOST – Document Intelligence API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS index & metadata on startup
load_index_and_metadata()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    filenames: List[str]
    top_k: int = 50
    model_name: str = "Claude 3.7 Sonnet"
    web_search: bool = False
    draft_mode: bool = False
    analyse_mode: bool = False
    eco_mode: bool = False

class EmailSeverityResponse(BaseModel):
    severity: str
    reasoning: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_page_ranges(filenames: List[str]) -> Dict[str, Tuple[int, int]]:
    ranges: Dict[str, Tuple[int, int]] = {}
    for fn in filenames:
        pages = [m["page"] for m in metadata_store if m["filename"] == fn]
        if not pages:
            raise HTTPException(status_code=404, detail=f"File '{fn}' not found in index")
        ranges[fn] = (min(pages), max(pages))
    return ranges


def _invoke_llm(system_msg: str, user_msg: str, model: str) -> str:
    if model == "Claude 3.5 Sonnet":
        return call_llm_api(system_msg, user_msg)
    if model == "Claude 3.7 Sonnet":
        return call_claude_api(system_msg, user_msg)
    if model == "GPT 4o":
        return call_gpt_api(system_msg, user_msg)
    if model == "Nova Lite":
        return call_novalite_api(system_msg, user_msg)
    if model == "Deepseek R1":
        return call_deepseek_api(system_msg, user_msg)
    return call_claude_api(system_msg, user_msg)

# ---------------------------------------------------------------------------
# Endpoint: list files
# ---------------------------------------------------------------------------

@app.get("/files", tags=["Documents"])
async def list_files(username: Optional[str] = Query(None, description="Filter by owner")):
    """Return list of filenames present in the index (optionally by owner)."""
    files = []
    for md in metadata_store:
        if username and md.get("owner") != username:
            continue
        files.append(md["filename"])
    unique_files = sorted(set(files))
    return {"count": len(unique_files), "files": unique_files}

# ---------------------------------------------------------------------------
# Endpoint: upload
# ---------------------------------------------------------------------------

@app.post("/upload", tags=["Documents"])
async def upload_files(
    files: List[UploadFile] = File(...),
    username: Optional[str] = Query(None, description="Username / owner of the files"),
):
    uploaded: List[str] = []
    for uf in files:
        try:
            wrapped = StreamlitFileAdapter(uf)
            add_file_to_index(wrapped)
            uploaded.append(uf.filename)
                        # Patch owner for every new metadata row for this file if `username` provided
            if username:
                for md in metadata_store:
                    if md["filename"] == uf.filename and md.get("owner") in (None, "unknown"):
                        md["owner"] = username
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed processing {uf.filename}: {exc}")

    # Persist to disk so other workers / future requests see the update
    save_index_and_metadata()
    return {"status": "success", "uploaded": uploaded}": uploaded}

# ---------------------------------------------------------------------------
# Endpoint: query
# ---------------------------------------------------------------------------

@app.post("/query", tags=["Documents"])
async def query_documents(req: QueryRequest):
    page_ranges = _full_page_ranges(req.filenames)
    chunks, answer, ws_data = query_documents_with_page_range(
        selected_files=req.filenames,
        selected_page_ranges=page_ranges,
        prompt=req.query,
        top_k=req.top_k,
        last_messages=[],
        web_search=req.web_search,
        llm_model=req.model_name,
        draft_mode=req.draft_mode,
        analyse_mode=req.analyse_mode,
        eco_mode=req.eco_mode,
    )
    return {"answer": answer, "sources": chunks, "web_search_results": ws_data}

# ---------------------------------------------------------------------------
# Endpoint: email severity
# ---------------------------------------------------------------------------

EMAIL_PROMPT = (
    "You are an email‑triage assistant for a busy legal‑tax consultancy. "
    "Read the email body plus any attached OCR text. "
    "Return JSON with keys `severity` (Low/Normal/Important/Urgent/Critical) and `reasoning`."
)

@app.post("/email-severity", response_model=EmailSeverityResponse, tags=["Email"])
async def email_severity(
    email_body: str = File(..., description="Raw email text body"),
    attachments: Optional[List[UploadFile]] = File(None, description="Email attachments"),
    model_name: str = "Claude 3.7 Sonnet",
):
    new_files: List[str] = []
    if attachments:
        for uf in attachments:
            wrapped = StreamlitFileAdapter(uf)
            add_file_to_index(wrapped)
            new_files.append(uf.filename)

    # gather attachment text
    blocks: List[str] = []
    for fn in new_files:
        pages = sorted([m for m in metadata_store if m["filename"] == fn], key=lambda x: x["page"])
        blocks.append(f"Attachment {fn}:\n" + "\n".join(p["text"] for p in pages))
    attach_txt = "\n\n".join(blocks) or "(no attachments)"

    llm_raw = _invoke_llm(EMAIL_PROMPT, f"EMAIL BODY:\n{email_body}\n\nATTACHMENTS:\n{attach_txt}", model_name)

    try:
        data = json.loads(llm_raw)
    except Exception:
        data = json.loads(re.search(r"\{.*\}", llm_raw, re.S).group(0))

    sev = str(data.get("severity", "Normal")).title()
    if sev.lower() not in {"low", "normal", "important", "urgent", "critical"}:
        sev = "Normal"
    reason = data.get("reasoning", "No reasoning provided.")
    return EmailSeverityResponse(severity=sev, reasoning=reason)

# ---------------------------------------------------------------------------
# Local dev entry‑point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
