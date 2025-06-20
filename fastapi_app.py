from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
import uvicorn
import asyncio

# --- Import helper logic directly from the Streamlit backend
# Note: fastapi_app.py is placed in the *same* folder as streamlit_app.py,
# so we can safely perform relative imports without modifying sys.path.
from app2 import (
    add_file_to_index,
    load_index_and_metadata,
    get_page_range_for_files,
    query_documents_with_page_range,
    metadata_store,
)

# Ensure the FAISS index & metadata are loaded when the service starts
load_index_and_metadata()

app = FastAPI(title="GST DOST – Document Intelligence API", version="1.0.0")

# Enable CORS so that the Streamlit frontend (or any browser client) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production lock this down
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# pydantic models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    filenames: List[str]
    top_k: int = 50
    model_name: str = "Claude 3.7 Sonnet"  # default model
    web_search: bool = False
    draft_mode: bool = False
    analyse_mode: bool = False
    eco_mode: bool = False

class EmailSeverityResponse(BaseModel):
    severity: str
    reasoning: str

# ---------------------------------------------------------------------------
# Endpoint 1: Upload & index documents
# ---------------------------------------------------------------------------
@app.post("/upload", tags=["Documents"])
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload one or more documents, store them, and add to the FAISS index."""
    uploaded = []
    for uf in files:
        try:
            # `add_file_to_index` consumes a *file‑like* object. We need to read
            # UploadFile into memory because the generator gets exhausted after
            # the first read when called inside the helper. We stream in chunks
            # to avoid big memory spikes.
            data = await uf.read()  # bytes
            uf.file.seek(0)
            add_file_to_index(uf)  # re‑use helper (handles all filetypes)
            uploaded.append(uf.filename)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed processing {uf.filename}: {exc}")
    return {"status": "success", "uploaded": uploaded}

# ---------------------------------------------------------------------------
# Internal helper for full‑range selection
# ---------------------------------------------------------------------------

def _build_full_page_range(filenames: List[str]) -> Dict[str, Tuple[int, int]]:
    """Return {filename: (min_page, max_page)} for every file in `filenames`."""
    ranges = {}
    for fn in filenames:
        pages = [m["page"] for m in metadata_store if m["filename"] == fn]
        if not pages:
            raise HTTPException(status_code=404, detail=f"File '{fn}' not found in index")
        ranges[fn] = (min(pages), max(pages))
    return ranges

# ---------------------------------------------------------------------------
# Endpoint 2: Query documents (full range – no page selection required)
# ---------------------------------------------------------------------------
@app.post("/query", tags=["Documents"])
async def query_documents(req: QueryRequest):
    # Build page ranges automatically (full doc) & call helper
    page_ranges = _build_full_page_range(req.filenames)

    # Use empty list for conversation context (stateless API) – callers can supply if needed
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
    return {
        "answer": answer,
        "sources": chunks,  # earliest  chunks list for transparency
        "web_search_results": ws_data,
    }

# ---------------------------------------------------------------------------
# Endpoint 3: Email‑severity classifier
# ---------------------------------------------------------------------------
EMAIL_CLASSIFICATION_PROMPT = """
You are an email‑triage assistant for a busy legal‑tax consultancy. Your task is to read the *email body* plus any *attached documents* (OCR‑extracted text below) and decide the urgency of the matter for internal escalation. 

Output a JSON with exactly two keys:\n  1. `severity` – one of: Low, Normal, Important, Urgent, Critical\n  2. `reasoning` – 1‑2 sentences explaining *why* you chose that severity.
"""

@app.post("/email-severity", response_model=EmailSeverityResponse, tags=["Email"])
async def email_severity(
    email_body: str = File(..., description="Raw email text body"),
    attachments: Optional[List[UploadFile]] = File(None, description="Email attachments (PDF, DOCX, etc.)"),
    model_name: str = "Claude 3.7 Sonnet",
):
    # 1) Index any attachments (if provided) – harmless to re‑index duplicates
    new_files = []
    if attachments:
        for uf in attachments:
            try:
                await uf.read()  # read once so helper can use file pointer again
                uf.file.seek(0)
                add_file_to_index(uf)
                new_files.append(uf.filename)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Attachment '{uf.filename}' failed: {exc}")

    # 2) Collect full‑range text from attachments (if any) for prompt context
    attachment_texts = []
    if new_files:
        for fn in new_files:
            pages = [m for m in metadata_store if m["filename"] == fn]
            pages_sorted = sorted(pages, key=lambda x: x["page"])
            joined_text = "\n".join(p["text"] for p in pages_sorted)
            attachment_texts.append(f"Attachment {fn}:\n{joined_text}")
    attachments_block = "\n\n".join(attachment_texts) if attachment_texts else "(no attachments)"

    # 3) Build final prompt & call chosen LLM (reuse call_selected_model logic):
    from streamlit_app import call_llm_api, call_claude_api, call_gpt_api, call_novalite_api, call_deepseek_api

    def _invoke_llm(system_msg: str, user_msg: str, mname: str):
        if mname == "Claude 3.5 Sonnet":
            return call_llm_api(system_msg, user_msg)
        if mname == "Claude 3.7 Sonnet":
            return call_claude_api(system_msg, user_msg)
        if mname == "GPT 4o":
            return call_gpt_api(system_msg, user_msg)
        if mname == "Nova Lite":
            return call_novalite_api(system_msg, user_msg)
        if mname == "Deepseek R1":
            return call_deepseek_api(system_msg, user_msg)
        # default
        return call_claude_api(system_msg, user_msg)

    system_msg = EMAIL_CLASSIFICATION_PROMPT
    user_msg = f"EMAIL BODY:\n{email_body}\n\nATTACHMENT TEXT:\n{attachments_block}"
    llm_raw = _invoke_llm(system_msg, user_msg, model_name)

    # 4) Parse JSON safely
    import json, re
    try:
        # Try direct JSON parse first
        data = json.loads(llm_raw)
    except Exception:
        # Extract JSON from markdown fence if present
        try:
            extracted = re.search(r"\{.*\}", llm_raw, re.S).group(0)
            data = json.loads(extracted)
        except Exception:
            raise HTTPException(status_code=500, detail="LLM response could not be parsed as JSON")

    # Basic validation
    sev_map = {"low", "normal", "important", "urgent", "critical"}
    severity_val = data.get("severity", "").strip().title()
    if severity_val.lower() not in sev_map:
        severity_val = "Normal"
    reasoning_val = data.get("reasoning", "No reasoning provided.")

    return EmailSeverityResponse(severity=severity_val, reasoning=reasoning_val)

# ---------------------------------------------------------------------------
# Optional script entry‑point for local dev:  `python fastapi_app.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
