from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
import uvicorn

"""fastapi_app.py

FastAPI micro‑service that re‑uses the heavy‑lifting helpers already defined in
`streamlit_app.py` (indexing, FAISS search, LLM calls).  The main difference
between Streamlit’s `UploadedFile` and FastAPI’s `UploadFile` is that the latter
exposes the filename through `.filename` instead of `.name`.  To bridge that
gap we wrap every incoming `UploadFile` in a lightweight adapter class that
adds the expected `.name` attribute so **all** helper routines continue to work
unchanged.
"""

# ---------------------------------------------------------------------------
# Imports & helper wrappers
# ---------------------------------------------------------------------------

# Standard / third‑party
import asyncio
import json
import re
from datetime import datetime

# FastAPI stack
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from typing import List, Dict, Tuple, Optional

# Helper logic pulled from Streamlit backend (same folder)
from app2 import (
    add_file_to_index,
    load_index_and_metadata,
    query_documents_with_page_range,
    metadata_store,
    call_llm_api,
    call_claude_api,
    call_gpt_api,
    call_novalite_api,
    call_deepseek_api,
)

# ---------------------------------------------------------------------------
# Adapter: make FastAPI UploadFile look like Streamlit UploadedFile
# ---------------------------------------------------------------------------

class StreamlitFileAdapter:  # noqa: N801 (keep camelCase like Streamlit)
    """Wrap `fastapi.UploadFile` so downstream helpers see `.name` & `.read`."""

    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename  # Streamlit uses `.name`
        self.filename = uf.filename  # convenience, keeps original attr too

    # Delegate typical file‑like methods
    def read(self, *args, **kwargs):
        return self._uf.file.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._uf.file.seek(*args, **kwargs)

    def __getattr__(self, item):  # fallback for anything else
        return getattr(self._uf, item)


# ---------------------------------------------------------------------------
# Initialise FastAPI & load index once at startup
# ---------------------------------------------------------------------------

app = FastAPI(title="GST DOST – Document Intelligence API", version="1.0.1")

# Allow cross‑origin requests (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bring FAISS and metadata into memory immediately
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
# Helper: full page‑range for each file
# ---------------------------------------------------------------------------

def _full_page_ranges(filenames: List[str]) -> Dict[str, Tuple[int, int]]:
    ranges: Dict[str, Tuple[int, int]] = {}
    for fn in filenames:
        pages = [m["page"] for m in metadata_store if m["filename"] == fn]
        if not pages:
            raise HTTPException(status_code=404, detail=f"File '{fn}' not found in index")
        ranges[fn] = (min(pages), max(pages))
    return ranges


# ---------------------------------------------------------------------------
# Endpoint 1 — upload & index
# ---------------------------------------------------------------------------

@app.post("/upload", tags=["Documents"])
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload one or more documents and add them to the shared FAISS index."""

    uploaded: List[str] = []
    for uf in files:
        try:
            wrapped = StreamlitFileAdapter(uf)
            add_file_to_index(wrapped)
            uploaded.append(uf.filename)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed processing {uf.filename}: {exc}") from exc
    return {"status": "success", "uploaded": uploaded}


# ---------------------------------------------------------------------------
# Endpoint 2 — query documents (auto full‑range)
# ---------------------------------------------------------------------------

@app.post("/query", tags=["Documents"])
async def query_documents(req: QueryRequest):
    page_ranges = _full_page_ranges(req.filenames)

    chunks, answer, ws_data = query_documents_with_page_range(
        selected_files=req.filenames,
        selected_page_ranges=page_ranges,
        prompt=req.query,
        top_k=req.top_k,
        last_messages=[],  # stateless
        web_search=req.web_search,
        llm_model=req.model_name,
        draft_mode=req.draft_mode,
        analyse_mode=req.analyse_mode,
        eco_mode=req.eco_mode,
    )

    return {"answer": answer, "sources": chunks, "web_search_results": ws_data}


# ---------------------------------------------------------------------------
# Endpoint 3 — email severity triage
# ---------------------------------------------------------------------------

EMAIL_PROMPT = (
    "You are an email‑triage assistant for a busy legal‑tax consultancy. "
    "Read the email body plus any attached OCR text. "
    "Return JSON with keys `severity` (Low/Normal/Important/Urgent/Critical) and `reasoning`."
)


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


@app.post("/email-severity", response_model=EmailSeverityResponse, tags=["Email"])
async def email_severity(
    email_body: str = File(..., description="Raw email text body"),
    attachments: Optional[List[UploadFile]] = File(None, description="Email attachments"),
    model_name: str = "Claude 3.7 Sonnet",
):

    # 1) index attachments (if provided)
    new_files: List[str] = []
    if attachments:
        for uf in attachments:
            try:
                wrapped = StreamlitFileAdapter(uf)
                add_file_to_index(wrapped)
                new_files.append(uf.filename)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Attachment '{uf.filename}' failed: {exc}") from exc

    # 2) gather attachment text
    attachment_text_blocks: List[str] = []
    for fn in new_files:
        pages = sorted((m for m in metadata_store if m["filename"] == fn), key=lambda x: x["page"])
        combined = "\n".join(p["text"] for p in pages)
        attachment_text_blocks.append(f"Attachment {fn}:\n{combined}")
    attachments_txt = "\n\n".join(attachment_text_blocks) or "(no attachments)"

    # 3) LLM call
    user_msg = f"EMAIL BODY:\n{email_body}\n\nATTACHMENTS:\n{attachments_txt}"
    raw = _invoke_llm(EMAIL_PROMPT, user_msg, model_name)

    # 4) parse JSON
    try:
        data = json.loads(raw)
    except Exception:
        try:
            data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        except Exception:
            raise HTTPException(status_code=500, detail="LLM response is not valid JSON")

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
