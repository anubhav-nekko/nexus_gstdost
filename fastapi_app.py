import os
import sys
import json
from typing import List, Dict, Tuple, Optional, Any
import re
import requests
import tempfile
import pickle
import fitz  # PyMuPDF
import faiss
import numpy as np
from datetime import datetime
import pytz
import boto3
from tavily import TavilyClient
from docx import Document
from pptx import Presentation
import pandas as pd
import io
from PIL import Image
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from openai import OpenAI
import argparse

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Body, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# --- 1. INITIAL SETUP AND CONFIGURATION ---

# Initialize FastAPI app
app = FastAPI(
    title="Nexus GSTDOST API",
    description="API for document management, querying, and email classification.",
    version="1.0.0"
)

# --- Pydantic Models for API Request/Response ---

class QueryRequest(BaseModel):
    username: str
    selected_files: List[str]
    query: str
    top_k: int = 50
    web_search: bool = False
    llm_model: str = "Claude 3.7 Sonnet"

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class EmailClassificationRequest(BaseModel):
    username: str
    file_name: str
    email_subject: str
    email_body: str
    trailing_emails_body: Optional[str] = None

class EmailClassificationResponse(BaseModel):
    classification: str
    reasoning: str

class FileListResponse(BaseModel):
    username: str
    files: List[str]

class UploadResponse(BaseModel):
    message: str
    filename: str
    user: str

# --- Global Variables ---
SECRETS = {}
USERS = {}
bedrock_runtime2 = None
textract_client = None
s3_client = None
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_STORE_PATH = "metadata_store.pkl"
dimension = 1024  # Cohere embedding dimension is 1024 for embed-multilingual-v3
faiss_index = None
metadata_store = []

# --- Configuration Loading ---
def load_dict_from_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Configuration file not found: {file_path}")

def initialize_config(secrets_path="secrets.json", users_path="users.json"):
    global SECRETS, USERS, bedrock_runtime2, textract_client, s3_client
    
    SECRETS = load_dict_from_json(secrets_path)
    USERS = load_dict_from_json(users_path)
    
    # AWS Credentials and other secrets
    aws_access_key_id = SECRETS["aws_access_key_id"]
    aws_secret_access_key = SECRETS["aws_secret_access_key"]
    REGION = SECRETS["REGION"]
    REGION2 = SECRETS["REGION2"]
    
    # Initialize AWS clients
    bedrock_runtime2 = boto3.client('bedrock-runtime', region_name=REGION2, 
                                   aws_access_key_id=aws_access_key_id, 
                                   aws_secret_access_key=aws_secret_access_key)
    textract_client = boto3.client('textract', region_name=REGION, 
                                  aws_access_key_id=aws_access_key_id, 
                                  aws_secret_access_key=aws_secret_access_key)
    s3_client = boto3.client('s3', region_name=REGION, 
                            aws_access_key_id=aws_access_key_id, 
                            aws_secret_access_key=aws_secret_access_key)

# --- Authentication ---
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key and any(pwd == api_key for pwd in USERS.values()):
        return api_key
    else:
        raise HTTPException(
            status_code=403, detail="Could not validate credentials"
        )

def get_user_from_key(api_key: str = Depends(get_api_key)):
    for user, pwd in USERS.items():
        if pwd == api_key:
            return user
    return None

# --- 2. CORE LOGIC AND HELPER FUNCTIONS ---

def call_claude_api(system_message, user_query):
    messages = [{"role": "user", "content": f"{system_message}\n\n{user_query}"}]
    payload = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 16384, "messages": messages}
    try:
        response = bedrock_runtime2.invoke_model(
            modelId=SECRETS["INFERENCE_PROFILE_CLAUDE"],
            contentType='application/json', accept='application/json', body=json.dumps(payload)
        )
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API call failed: {str(e)}")

def save_index_and_metadata():
    global faiss_index, metadata_store
    try:
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        with open(METADATA_STORE_PATH, "wb") as f:
            pickle.dump(metadata_store, f)
        
        s3_client.upload_file(FAISS_INDEX_PATH, SECRETS["s3_bucket_name"], os.path.basename(FAISS_INDEX_PATH))
        s3_client.upload_file(METADATA_STORE_PATH, SECRETS["s3_bucket_name"], os.path.basename(METADATA_STORE_PATH))
        print("Index and metadata saved to S3.")
    except Exception as e:
        print(f"Error saving index and metadata: {e}")

def load_index_and_metadata():
    global faiss_index, metadata_store
    index_blob_name = os.path.basename(FAISS_INDEX_PATH)
    metadata_blob_name = os.path.basename(METADATA_STORE_PATH)

    try:
        s3_client.download_file(SECRETS["s3_bucket_name"], index_blob_name, FAISS_INDEX_PATH)
        s3_client.download_file(SECRETS["s3_bucket_name"], metadata_blob_name, METADATA_STORE_PATH)
        
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_STORE_PATH, "rb") as f:
            metadata_store = pickle.load(f)
        print("Index and metadata loaded from S3.")
    except Exception as e:
        print(f"Index or metadata not found in S3. Initializing new. Error: {e}")
        faiss_index = faiss.IndexFlatL2(dimension)
        metadata_store = []

def generate_titan_embeddings(text):
    """
    Generate embeddings for given text(s) using Cohere via Bedrock (`bedrock_runtime2`).
    Truncates each text to 2048 characters as per model limits and returns a NumPy array.
    
    Args:
        text (str or list of str): Input text or list of texts to embed.
        
    Returns:
        np.ndarray or None: 
            - If `text` is a single string, returns a 1D array of shape (embedding_dim,).
            - If `text` is a list of strings, returns a 2D array of shape (len(texts), embedding_dim).
            - Returns None on failure.
    """
    try:
        # Detect whether user passed a single string or a list of strings
        single_input = False
        if isinstance(text, str):
            texts = [text[:2048]]
            single_input = True
        else:
            texts = [t[:2048] for t in text]

        # Build the JSON payload exactly as Cohere expects
        body = {
            "texts": texts,
            "input_type": "search_document"
        }

        # Invoke the Cohere model via your existing bedrock-runtime client
        response = bedrock_runtime2.invoke_model(
            modelId="cohere.embed-multilingual-v3",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )

        # Bedrock Runtime returns a StreamingBody under response["body"]
        resp_body = response.get("body")
        if resp_body is None:
            print("❌ No 'body' in Bedrock Runtime response:", response)
            return None

        # Read & decode the streaming response, then parse JSON
        decoded = resp_body.read().decode("utf-8")
        resp_json = json.loads(decoded)

        # Extract embeddings
        embeddings = resp_json.get("embeddings")
        if embeddings is None:
            print("⚠️ 'embeddings' key missing in response:", resp_json)
            return None

        # Convert to NumPy array
        arr = np.array(embeddings, dtype=np.float32)

        # If a single string was given, return the first (and only) vector
        return arr[0] if single_input else arr

    except Exception as e:
        print(f"❌ Error generating embeddings from Cohere via Bedrock Runtime: {e}")
        return None

def extract_text_from_pdf_bytes(file_bytes: bytes) -> List[Tuple[int, str]]:
    pages_text = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            
            response = textract_client.detect_document_text(Document={'Bytes': img_bytes})
            
            text_lines = [block['Text'] for block in response.get('Blocks', []) if block['BlockType'] == 'LINE' and 'Text' in block]
            page_text = "\n".join(text_lines)
            pages_text.append((page_num + 1, page_text))
        doc.close()
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return pages_text

def add_file_to_index(file_bytes: bytes, filename: str, username: str):
    global faiss_index, metadata_store
    
    if any(md['filename'] == filename and md['owner'] == username for md in metadata_store):
        raise HTTPException(status_code=409, detail=f"File '{filename}' already exists for user '{username}'.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        s3_client.upload_file(temp_file_path, SECRETS["s3_bucket_name"], filename)

        pages_text_tuples = extract_text_from_pdf_bytes(file_bytes)
        
        for page_num, text in pages_text_tuples:
            if text.strip():  # Only process non-empty text
                embedding = generate_titan_embeddings(text)
                if embedding is not None:
                    faiss_index.add(embedding.reshape(1, -1))
                    metadata_store.append({
                        "filename": filename,
                        "page": page_num,
                        "text": text,
                        "owner": username,
                        "shared_with": []
                    })
        
        save_index_and_metadata()

    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def get_page_range_for_files(selected_files: List[str], username: str) -> Dict[str, Tuple[int, int]]:
    page_ranges = {}
    for file in selected_files:
        # Ensure the user has access to the file
        file_metadata = [md for md in metadata_store if md['filename'] == file and (md['owner'] == username or username in md.get('shared_with', []))]
        if file_metadata:
            min_page = min(md['page'] for md in file_metadata)
            max_page = max(md['page'] for md in file_metadata)
            page_ranges[file] = (min_page, max_page)
    return page_ranges

def execute_query(request: QueryRequest) -> QueryResponse:
    if faiss_index.ntotal == 0:
        raise HTTPException(status_code=404, detail="No documents have been indexed yet.")

    query_embedding = generate_titan_embeddings(request.query)
    if query_embedding is None:
        raise HTTPException(status_code=500, detail="Failed to generate embeddings for the query.")
    query_embedding = query_embedding.reshape(1, -1)
    
    selected_page_ranges = get_page_range_for_files(request.selected_files, request.username)

    k = faiss_index.ntotal
    distances, indices = faiss_index.search(query_embedding, k)
    
    filtered_results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata_store):
            metadata = metadata_store[idx]
            # Filter by selected files AND user access (owner or shared_with)
            if metadata['filename'] in request.selected_files and (metadata['owner'] == request.username or request.username in metadata.get('shared_with', [])):
                min_page, max_page = selected_page_ranges.get(metadata['filename'], (None, None))
                if min_page is not None and min_page <= metadata['page'] <= max_page:
                    filtered_results.append((dist, idx))
    
    top_k_results = sorted(filtered_results, key=lambda x: x[0])[:request.top_k]
    top_k_metadata = [metadata_store[idx] for _, idx in top_k_results]

    if not top_k_metadata:
        return QueryResponse(answer="No relevant information found in the selected documents for your query.", sources=[])

    # Create a more focused context for the LLM
    context_texts = []
    for metadata in top_k_metadata:
        context_texts.append(f"File: {metadata['filename']}, Page: {metadata['page']}\nContent: {metadata['text'][:1000]}...")
    
    context = "\n\n".join(context_texts)
    system_prompt = """You are a helpful AI assistant. Based on the provided context from documents, answer the user's query. 
    Be concise and accurate. If the information is not available in the context, say so clearly.
    Always cite the source file and page number when providing information."""
    user_prompt = f"Context:\n{context}\n\nQuery: {request.query}"
    
    answer = call_claude_api(system_prompt, user_prompt)
    
    return QueryResponse(answer=answer, sources=top_k_metadata)

# --- 3. API ENDPOINTS ---

@app.post("/upload-document/", response_model=UploadResponse, dependencies=[Depends(get_api_key)])
async def upload_document(file: UploadFile = File(...), username: str = Depends(get_user_from_key)):
    if not file.filename.lower().endswith('.pdf'):
         raise HTTPException(status_code=400, detail="Currently, only PDF files are supported.")

    file_bytes = await file.read()
    add_file_to_index(file_bytes, file.filename, username)
    
    return UploadResponse(
        message="File uploaded and processed successfully.",
        filename=file.filename,
        user=username
    )

@app.post("/query-documents/", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    if request.username not in USERS:
        raise HTTPException(status_code=404, detail=f"User '{request.username}' not found.")
        
    return execute_query(request)

@app.post("/classify-email/", response_model=EmailClassificationResponse)
def classify_email(request: EmailClassificationRequest):
    if request.username not in USERS:
        raise HTTPException(status_code=404, detail=f"User '{request.username}' not found.")

    attachment_text = ""
    # Ensure the user has access to the file
    attachment_metadata = [
        md['text'] for md in metadata_store 
        if md['filename'] == request.file_name and (md['owner'] == request.username or request.username in md.get('shared_with', []))
    ]
    if not attachment_metadata:
        raise HTTPException(status_code=404, detail=f"Attachment '{request.file_name}' not found or not accessible for user '{request.username}'.")
    
    attachment_text = "\n".join(attachment_metadata)

    full_email_context = f"Email Subject: {request.email_subject}\nEmail Body: {request.email_body}\n"
    if request.trailing_emails_body:
        full_email_context += f"Trailing Emails Body:\n{request.trailing_emails_body}\n"
    
    full_email_context += f"\n--- Attached Document Content ---\n{attachment_text[:5000]}..."  # Limit context size

    system_prompt = """
    You are an expert email analysis AI. Your task is to classify an email into one of four categories:
    - Critical: Requires immediate action, high urgency, significant financial or legal implications.
    - Important: Requires action soon, has significant business relevance but is not immediately critical.
    - Standard: Routine communication, informational, no urgent action required.
    - Spam/Junk: Unsolicited, irrelevant, or promotional content.

    Analyze the provided email content (subject, body, and attachment) and respond with a JSON object containing two keys: 'classification' and 'reasoning'.
    """
    user_prompt = f"Please classify the following email:\n\n{full_email_context}"

    response_text = call_claude_api(system_prompt, user_prompt)

    try:
        # Attempt to extract JSON from the response, handling common LLM formatting
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()
        
        result = json.loads(json_str)
        return EmailClassificationResponse(
            classification=result.get("classification", "Unclassified"),
            reasoning=result.get("reasoning", "No reasoning provided.")
        )
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        # Fallback parsing
        classification = "Standard"
        reasoning = "Unable to parse classification response properly."
        
        if "critical" in response_text.lower():
            classification = "Critical"
        elif "important" in response_text.lower():
            classification = "Important"
        elif "spam" in response_text.lower() or "junk" in response_text.lower():
            classification = "Spam/Junk"
            
        return EmailClassificationResponse(
            classification=classification,
            reasoning=f"Fallback classification. Original response: {response_text[:200]}..."
        )

@app.get("/list-files/{username}", response_model=FileListResponse)
def list_files(username: str):
    if username not in USERS:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found.")

    user_files = {
        md["filename"]
        for md in metadata_store
        if md.get("owner") == username or username in md.get("shared_with", [])
    }
    
    return FileListResponse(username=username, files=sorted(list(user_files)))

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is running"}

# --- 4. STARTUP AND MAIN ---

@app.on_event("startup")
def startup_event():
    # Configuration will be initialized when the app starts
    pass

def main():
    parser = argparse.ArgumentParser(description='Run Nexus GSTDOST FastAPI server')
    parser.add_argument('--secrets', default='secrets.json', help='Path to secrets.json file')
    parser.add_argument('--users', default='users.json', help='Path to users.json file')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    
    args = parser.parse_args()
    
    # Initialize configuration
    initialize_config(args.secrets, args.users)
    load_index_and_metadata()
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

