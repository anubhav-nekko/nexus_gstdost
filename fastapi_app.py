from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
import uvicorn
import json, re

"""fastapi_app.py - GST DOST backend micro-service

Key endpoints

1. **POST /upload** -- store & index files. Accepts optional `username` so we can tag
   ownership in `metadata_store` even though `streamlit_app.add_file_to_index` was built
   for Streamlit sessions.
2. **GET /files** -- list all indexed files (optionally filter by `username`).
3. **POST /query** -- natural-language search across full documents (no page range UI).
4. **POST /email-severity** -- classify an email + attachments into Low / Normal / ...

The helper functions come directly from `streamlit_app.py` (same folder). The only
incompatibility is FastAPI's `UploadFile` exposing `.filename` instead of `.name`, which
we solve via the `StreamlitFileAdapter` wrapper.
"""

# Imports & helper wrappers
# #########################

from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional

# Helper logic from Streamlit backend
from fastapihelper import (
    add_file_to_index,
    load_index_and_metadata,
    query_documents_with_page_range,
    metadata_store,
    save_index_and_metadata,
    call_llm_api,
    call_claude_api,
    call_gpt_api,
    call_novalite_api,
    call_deepseek_api,
)


# FastAPI app setup
# #################

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Endpoints
# #########

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    username: Optional[str] = None,
    # metadata: Optional[str] = None, # for future use
):
    try:
        # Add file to index using the helper function
        add_file_to_index(file, username)

        # Save index and metadata (ensure persistence)
        save_index_and_metadata()

        return {"message": f"File '{file.filename}' uploaded and indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload and index file: {e}")


@app.get("/files")
async def list_files(username: Optional[str] = None):
    try:
        # Load index and metadata
        load_index_and_metadata()

        # Filter files by username if provided
        if username:
            user_files = [f for f in metadata_store.keys() if metadata_store[f].get("username") == username]
            return {"files": user_files}
        else:
            return {"files": list(metadata_store.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {e}")


@app.post("/query")
async def query_documents(
    query: str,
    username: Optional[str] = None,
    page_range: Optional[str] = None,  # e.g., "1-5, 7, 10-12"
):
    try:
        # Load index and metadata
        load_index_and_metadata()

        # Perform query using the helper function
        results = query_documents_with_page_range(query, username, page_range)

        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query documents: {e}")


@app.post("/email-severity")
async def email_severity(
    email_content: str,
    attachments: List[UploadFile] = File([]),
    username: Optional[str] = None,
):
    try:
        # Placeholder for email severity classification logic
        # This would involve calling an LLM or other classification model
        # For now, just return a dummy response

        # Example of how you might process attachments:
        attachment_names = [att.filename for att in attachments]

        # Call LLM API for classification
        # For demonstration, let's assume a simple classification based on keywords
        if "urgent" in email_content.lower() or "action required" in email_content.lower():
            severity = "High"
        elif "meeting" in email_content.lower() or "info" in email_content.lower():
            severity = "Normal"
        else:
            severity = "Low"

        return {
            "email_content": email_content,
            "attachments": attachment_names,
            "severity": severity,
            "message": "Email classified successfully (dummy classification)",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to classify email severity: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


