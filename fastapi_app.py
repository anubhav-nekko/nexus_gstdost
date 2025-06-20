# fastapi_app.py
"""Nexus DMS – FastAPI micro‑service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A thin API wrapper that re‑uses the heavy‑lifting utilities living in
`streamlit_app.py`.  It lets other services (n8n / Zapier / CRON / mobile
clients) ingest new evidence and run queries without going through the
Streamlit front‑end.

Key additions requested
----------------------
1. **`GET /files`** – list every file currently indexed (owner + page list).
2. **Username segregation** – the `POST /upload-files` endpoint now accepts a
   `username` form‑field; we temporarily set `st.session_state["username"]`
   so helper functions can record ownership.

The existing three endpoints are retained:
* **`POST /upload-files`** – upload & index one or many files.
* **`POST /query`** – semantic search across selected filenames (full page
  range automatically detected).
* **`POST /email-severity`** – classify an email body (plus optional
  attachment) into *Normal / Important / Critical* with JSON rationale.

Because `streamlit_app.py` relies on the global `streamlit` session state, we
import `streamlit as st` and prime `st.session_state` so that helper functions
continue to work even outside a Streamlit context.
"""
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List, Dict, Tuple
import json, re

# ── Re‑use everything we already built in the Streamlit layer ────────────────
import streamlit as st                  # noqa – needed for helper functions
import app2 as sa             # the big helper module you showed

# Ensure the session_state keys exist so helper funcs don’t crash
if "username" not in st.session_state:
    st.session_state["username"] = "api_user"

# ---------------------------------------------------------------------------
app = FastAPI(title="Nexus DMS API", version="1.1.0")

# ── Utility helpers ─────────────────────────────────────────────────────────

def _patch_file(f: UploadFile) -> UploadFile:
    """Monkey‑patch FastAPI `UploadFile` so it looks like a Streamlit file."""
    if not hasattr(f, "name"):
        f.name = f.filename            # Streamlit’s helper expects `.name`
    return f


def _full_page_ranges(filenames: List[str]) -> Dict[str, Tuple[int, int]]:
    """Return {filename: (min_pg, max_pg)} using sa.metadata_store."""
    ranges: Dict[str, Tuple[int, int]] = {}
    for fname in filenames:
        pages = [m["page"] for m in sa.metadata_store if m["filename"] == fname]
        if not pages:
            raise HTTPException(404, f"No pages indexed for {fname}")
        ranges[fname] = (min(pages), max(pages))
    return ranges

# ── API End‑points ──────────────────────────────────────────────────────────

@app.post("/upload-files")
async def upload_files(
    username: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Upload one or many documents, index them and tag with *owner=username*."""
    st.session_state["username"] = username  # hand‑off for helper functions

    saved: List[str] = []
    for up in files:
        f = _patch_file(up)
        try:
            sa.add_file_to_index(f)   # uses username from session_state
            saved.append(f.name)
        except Exception as e:        # pragma: no cover
            raise HTTPException(500, f"Failed processing {f.name}: {e}")
    return {"status": "success", "files_saved": saved}


@app.get("/files", response_model=Dict[str, Dict])
def list_files():
    """Return every unique filename currently present together with owner & pages."""
    registry: Dict[str, Dict] = {}
    for m in sa.metadata_store:
        fname, owner, pg = m["filename"], m.get("owner", "unknown"), m["page"]
        entry = registry.setdefault(fname, {"owner": owner, "pages": set()})
        entry["pages"].add(pg)
    # convert page sets to sorted lists so FastAPI can serialise them
    for e in registry.values():
        e["pages"] = sorted(e["pages"])
    return registry


@app.post("/query")
async def query_documents(
    query: str = Form(...),
    filenames: List[str] = Form(...),
    top_k: int = Form(20)
):
    """Run a semantic query across selected filenames (full page range)."""
    selected_files = list(filenames)
    selected_page_ranges = _full_page_ranges(selected_files)

    top_md, answer, _ = sa.query_documents_with_page_range(
        selected_files,
        selected_page_ranges,
        query,
        top_k,
        last_messages=[],
        web_search=False,
        llm_model="Claude 3.5 Sonnet",
        draft_mode=False,
        analyse_mode=False,
        eco_mode=False,
    )
    return {"answer": answer, "sources": top_md}


@app.post("/email-severity")
async def email_severity(
    username: str = Form(...),
    email_body: str = Form(...),
    attachment: UploadFile | None = File(None)
):
    """Classify an incoming email + attachment as Normal/Important/Critical."""
    st.session_state["username"] = username

    attachment_text = ""
    attach_name: str | None = None
    if attachment is not None:
        f = _patch_file(attachment)
        sa.add_file_to_index(f)
        attach_name = f.name
        # gather all text from that file for context (may be large)
        attachment_text = "\n".join(
            m["text"] for m in sa.metadata_store if m["filename"] == f.name
        )[:4000]  # cap to keep prompt size sane

    prompt = (
        "You are a compliance e‑mail triage bot. Categories: Normal, Important, "
        "Critical. Read the e‑mail and optional attachment excerpt. Return *only* "
        "valid JSON with keys `severity` and `reasoning`.\n\n"
        f"Email Body:\n{email_body}\n\nAttachment Excerpt:\n{attachment_text}"
    )

    raw = sa.call_llm_api("You are a JSON‑only responder.", prompt)
    # attempt to parse; fall back gracefully
    try:
        if "```" in raw:
            raw = re.split(r"```(?:json)?", raw)[1]
        result = json.loads(raw)
    except Exception:
        result = {"severity": "Normal", "reasoning": raw.strip()[:500]}

    # add a bit of meta for caller convenience
    result.update(
        {
            "attachment_processed": attachment is not None,
            "attachment_filename": attach_name,
        }
    )
    return result

# ---------------------------------------------------------------------------
# Optional: a simple health‑check
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
