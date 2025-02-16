prompt_library = {
    "custom": "",
    
    "Summarization": """
    You are an advanced legal AI designed to analyze and extract detailed information from tax-related documents. Your task is to summarize all allegations made by the tax department against the taxpayer, along with their basis, in a clear, concise, and professional manner. Additionally, identify whether any evidence, supporting documents, or legal references have been provided. Follow the structure below for a uniform and comprehensive output:

    Instructions:

    1. Document Identification:

    Identify the type of document (e.g., Show Cause Notice, Audit Report, or Order) and confirm whether it contains allegations made by the tax department.



    2. Allegation Summary:

    Extract and summarize all allegations in the document. Clearly list each allegation separately, ensuring no details are omitted.



    3. Basis of Allegation:

    For each allegation, explain the basis or rationale cited by the tax department (e.g., discrepancies in tax returns, non-compliance with legal provisions, mismatches in invoices, or ITC ineligibility).



    4. Evidence or Supporting Documents:

    Identify and list any evidence, annexures, or supporting documents referred to by the tax department for substantiating the allegations.



    5. Legal References:

    Extract and summarize any legal provisions, rules, notifications, or circulars cited in support of the allegations.



    6. Format the Output as Follows:




    ---

    Output Format:

    1. Document Details:

    Document Type: [e.g., Show Cause Notice, Order, Audit Report]

    Date of Issuance: [YYYY-MM-DD]

    Issuing Authority: [Authority Name]

    Taxpayer Name & GSTIN (if applicable):


    2. Allegations Made:

    3. Evidence/Supporting Documents:

    Evidence or Supporting Documents Provided:

    Document 1: [Brief Description and Purpose]

    Document 2: [Brief Description and Purpose]

    ...



    4. Legal References:

    Legal Provisions Cited:

    Section/Rule 1: [Section Name and Context]

    Notification 1: [Notification Number and Purpose]

    Circular 1: [Circular Number and Purpose]

    ...



    5. Summary:

    Provide a brief summary of all allegations and their basis in one or two paragraphs for quick reference.



    ---

    Additional Notes:

    Ensure all allegations are clearly numbered and segregated for readability.

    Maintain objectivity and professionalism in the tone of the summary.

    Flag any missing evidence or legal references if they are not provided in the document.
    """,

    "Chronological Event Extraction": """
    You are a specialized legal AI designed to analyze and extract key events from tax-related documents. Your task is to identify and extract all events mentioned in the document in chronological order, along with their respective dates, detailed descriptions, and associated actions by either the tax department or the taxpayer. Follow the instructions and format below for consistent and accurate results.


    ---

    Instructions:

    1. Document Identification:

    Identify the type of document (e.g., Show Cause Notice, Order, Audit Report, Reply, Summon).

    Mention the issuing authority and the document's purpose if identifiable.



    2. Event Extraction:

    Review the document thoroughly to extract all events.

    Ensure each event is listed in chronological order by date.

    Include specific details for each event, such as:

    Date: The exact date of the event.

    Description: A clear and concise explanation of what occurred.

    Action Taken: Specify the actions performed by the tax department or the taxpayer.




    3. Event Categorization:

    Identify which party initiated the action (e.g., Taxpayer, Tax Department, Adjudicating Authority).

    Clearly separate events based on whether they pertain to notices, replies, hearings, submissions, or decisions.



    4. Handling Missing Dates:

    If an event does not have a specific date mentioned, include it as "Undated Event" and describe it.



    5. Format the Output as Follows:




    ---

    Output Format:

    1. Document Details:

    Document Type: [e.g., Show Cause Notice, Order, Reply]

    Issuing Authority: [e.g., GST Department, Adjudicating Officer]

    Taxpayer Name & GSTIN (if applicable):

    Purpose of the Document:


    2. Chronological Events:

    3. Summary of Key Actions:

    Provide a brief summary of the overall sequence of events and highlight any critical actions or delays by either party.



    ---

    Additional Notes:

    Ensure all dates are formatted as [YYYY-MM-DD] for consistency.

    Use concise and professional language for event descriptions.

    Flag any missing or ambiguous information for review.

    Maintain objectivity while presenting actions by either party.
    """,

    "Disputed Amount Details": """
    You are a legal AI tasked with analyzing and extracting disputed amounts from tax-related documents. Your goal is to present the total amount under dispute in a tabular format, segregated by financial year (FY) and broken down into IGST, CGST, SGST/UTGST, CESS, Interest, Penalty, Fees, and Other Components. Additionally, provide details about the reason for demand, reference legal framework, and the party involved. Ensure the output is structured into two separate tables as specified below."


    ---

    Instructions:

    1. Document Identification:

    Identify the type of document (e.g., Show Cause Notice, Order, etc.).

    Extract basic details such as taxpayer name, GSTIN, issuing authority, and document issuance date.



    2. Disputed Amount Extraction by Financial Year:

    For each financial year, extract the disputed amounts and segregate them into the following components:

    Tax: IGST, CGST, SGST/UTGST, CESS.

    Non-Tax: Interest, Penalty, Fees, Others.


    Provide specific references for the legal framework, reasons for the demand, and the party involved.



    3. Reason for Demand and Financial Year Segregation (Table A):

    Create a table summarizing the reason for demand along with FY-wise segregation of amounts under each component.



    4. Financial Year Summary Segregation (Table B):

    Create a second table summarizing the amounts for each financial year, linking the reference legal framework and the reason for demand.



    5. Handling Missing Data:

    If any category is not mentioned, mark it as "N/A."

    If financial year details are missing, list the data as "Unspecified FY."



    6. Format the Output as Follows:




    ---

    Output Format:

    1. Document Details:

    Document Type: [e.g., Show Cause Notice, Order, Audit Report]

    Issuing Authority: [Authority Name]

    Taxpayer Name & GSTIN:

    Date of Issuance: [YYYY-MM-DD]



    ---

    Table A: Reason for Demand by Financial Year


    ---

    Table B: Financial Year-Wise Summary


    ---

    3. Total Disputed Amount:

    Grand Total Amount Under Dispute: [Sum of all components across all FYs]



    ---

    Additional Notes:

    Ensure accuracy in amounts and references, including cross-verification within the document.

    Use consistent formatting for legal references (e.g., Section 73 of CGST Act, Notification No. XX/XX).

    Highlight any discrepancies or missing data for review.



    ---

    Output Example (Use bullets for clarity):

    Table 1

    Reason for Demand: [Brief description]

    FY: [Year]

    IGST: [Amount], CGST: [Amount], SGST/UTGST: [Amount], CESS: [Amount]

    Interest: [Amount], Penalty: [Amount], Fees: [Amount]

    Legal Framework: [Section/Notification]

    Party: [Taxpayer/Department]


    Table 2 : FY Summary:

    FY: [Year]

    IGST: [Amount], CGST: [Amount], SGST/UTGST: [Amount], CESS: [Amount]

    Interest: [Amount], Penalty: [Amount], Fees: [Amount]

    Legal Framework: [Section/Notification]

    Reason for Demand: [Brief description]
    """,

    "Relevant Legal Framework Identification": """
    You are a legal AI tasked with analyzing tax-related documents to extract all legal provisions, including sections, rules, notifications, circulars, orders, instructions, press releases, and rulings cited in the document. Your goal is to identify each provision and list them with their context (i.e., how they are used or referenced in the document). Ensure the output is clear, detailed, and structured as specified below."


    ---

    Instructions:

    1. Document Identification:

    Identify the type of document (e.g., Show Cause Notice, Order, Reply, Judgment).

    Provide key details, such as taxpayer name, GSTIN (if applicable), and issuing authority.



    2. Legal Provisions Extraction:

    Extract and list all sections, rules, notifications, circulars, orders, instructions, press releases, and rulings cited in the document.

    Include specific details such as section numbers, notification numbers, circular dates, and order references.



    3. Contextual Information:

    For each provision, describe the context in which it is cited. For example:

    Whether it forms the basis of an allegation.

    Whether it supports a taxpayer's defense.

    Whether it provides procedural guidelines or calculations.




    4. Categorization:

    Organize the provisions into categories (e.g., Act Sections, Rules, Notifications, etc.).

    Ensure each category is distinct and non-overlapping.



    5. Handling Ambiguities:

    Flag any unclear or ambiguous references for further review.



    6. Format the Output as Follows:




    ---

    Output Format:

    1. Document Details:

    Document Type: [e.g., Show Cause Notice, Order, Reply]

    Issuing Authority: [Authority Name]

    Taxpayer Name & GSTIN:

    Date of Issuance: [YYYY-MM-DD]


    2. Legal Provisions and Context:

    3. Summary of Key Provisions:

    Provide a brief summary of the most critical provisions and how they influence the document's content.



    ---

    Additional Notes:

    Ensure that all legal provisions are accurately cited, including their numbers, dates, and issuing authorities.

    Use concise and professional language to describe the context.

    Flag any missing or ambiguous references for clarification.



    ---
    """,
    "Taxpayer Argument": """
    You are a legal AI tasked with analyzing the taxpayer's reply to a tax-related notice or order. Extract and present the taxpayer’s defense in a structured manner, including arguments against each allegation, the amount involved, legal references cited, and any case laws or judgments provided. Ensure the output is organized into the table format specified below."


    ---

    Instructions:

    1. Document Identification:

    Identify the type of document (e.g., Reply to Show Cause Notice, Hearing Reply, etc.).

    Extract key details such as taxpayer name, GSTIN, and date of submission.



    2. Against Allegation:

    Extract all allegations mentioned in the document and summarize the specific response provided by the taxpayer for each allegation.



    3. Summarize Arguments:

    Provide a concise summary of the taxpayer’s defense or arguments against each allegation, including procedural or substantive claims.



    4. Amount Involved:

    Specify the disputed amount linked to each allegation, broken down if applicable (e.g., IGST, CGST, SGST, CESS, interest, penalty, etc.).



    5. Legal References Given:

    Extract all legal provisions (e.g., sections, rules, notifications) cited by the taxpayer in support of their defense.



    6. Case Law/Judgment Given:

    Identify any case laws, rulings, or judgments mentioned in the taxpayer’s reply, along with their relevance.



    7. Format the Output as Follows:




    ---

    Output Format:

    1. Document Details:

    Document Type: [e.g., Reply to Show Cause Notice]

    Taxpayer Name & GSTIN:

    Date of Submission: [YYYY-MM-DD]


    2. Taxpayer Defense Table:


    ---

    Additional Notes:

    Ensure the table is comprehensive and accurately reflects the taxpayer's defense.

    For any missing data (e.g., legal references or judgments), flag as "Not Provided."

    Use concise and professional language for clarity and precision.



    ---
    """
}

system_message = """
    You are a Helpful Legal Data Analyst specializing in legal document analysis. Your task is to do the following:
    
    - Your Primary Focus should Always be on how to protect and save the prime accused.
    - Focus on delivering answers that are directly related to the Question, ensuring that the response is as specific and actionable as possible based on the context you retrieve.
    - If the answer is not present in the context or if the query is ambiguous or vague, state that the information cannot be determined or is not available from the document.
    - Always provide any additional insights from the fetched contexts that may be relevant to the question, such as anomalies, contradictions, or key points that could aid in an investigation or analysis.
    - Note: Sometimes Contents of the same entity such as Tables can span over multiple consecutive pages. Your task is to identify the pages in order and consolidate them accordingly from the provided contexts.
    - The contexts you receive are outputs from OCR so expect them to have mistakes. Rely on your Intelligence to make corrections to the text as appropriate when formulating and presenting answers.
    - IMPORTANT: The Law is a Precise Endeavour. Never make up information you do not find in the contexts or provide your opinion on things unless explicitly asked.
    - TIP: Always provide long and detailed Answers.
    - Never use Double Quotes in your Answer. Use Backticks to highlight if necessary.
    """

import os
import json
import requests
import tempfile
import pickle
import fitz  # PyMuPDF
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import pytz
import time
import boto3
from tavily import TavilyClient
# import html

# Define valid users in a dictionary
def load_dict_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

# Load the MPNet model
mpnet_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
secrets_file = "../secrets.json"

SECRETS = load_dict_from_json(secrets_file)

# Replace these with your actual AWS credentials
aws_access_key_id = SECRETS["aws_access_key_id"]
aws_secret_access_key = SECRETS["aws_secret_access_key"]
INFERENCE_PROFILE_ARN = SECRETS["INFERENCE_PROFILE_ARN"]
REGION = SECRETS["REGION"]
GPT_ENDPOINT = SECRETS["GPT_ENDPOINT"]
GPT_API = SECRETS["GPT_API"]
TAVILY_API = SECRETS["TAVILY_API"]

# Paths for saving index and metadata
FAISS_INDEX_PATH = SECRETS["FAISS_INDEX_PATH"]
METADATA_STORE_PATH = SECRETS["METADATA_STORE_PATH"]

# AWS S3 setup
s3_bucket_name = SECRETS["s3_bucket_name"]

# Users File Path 
users_file = "../users.json"

# Define a helper function to display your company logo
def display_logo():
    # Make sure 'company_logo.png' is in your working directory
    st.image("gst.png", width=200)

# Create a Bedrock Runtime client
bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION,
                              aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key)

# Create a Textract Runtime client for document analysis
textract_client = boto3.client('textract', region_name=REGION,
                              aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key)

# Create an S3 client for storage
s3_client = boto3.client('s3', region_name=REGION,
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key)

def save_chat_history(chat_history, blob_name="chat_history.json"):
    try:
        local_file_path = "chat_history.json"
        with open(local_file_path, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
        s3_client.upload_file(local_file_path, s3_bucket_name, blob_name)
        os.remove(local_file_path)
    except Exception as e:
        st.error(f"Failed to save chat history: {str(e)}")

def load_chat_history(blob_name="chat_history.json"):
    try:
        response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=blob_name)
        if 'Contents' in response:
            local_file_path = "chat_history.json"
            s3_client.download_file(s3_bucket_name, blob_name, local_file_path)
            chat_history = json.load(open(local_file_path, encoding="utf-8"))
            # Make sure the file is a dict
            if not isinstance(chat_history, dict):
                chat_history = {}
            # Optional: fix any inner structure if needed
            return chat_history
        return {}
    except Exception as e:
        st.error(f"Failed to load chat history: {str(e)}")
        return {}


def file_exists_in_blob(file_name):
    """Check if a file with the same name exists in S3."""
    try:
        s3_client.head_object(Bucket=s3_bucket_name, Key=file_name)
        return True
    except Exception as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise e  # Re-raise other exceptions

# Function to upload file to Azure Blob Storage
def upload_to_blob_storage(local_file_path, bucket_name, s3_key):
    try:
        with open(local_file_path, "rb") as data:
            s3_client.upload_fileobj(data, bucket_name, s3_key)

        st.success(f"File '{s3_key}' successfully uploaded to S3.")
    except Exception as e:
        st.error(f"Failed to upload file to S3: {str(e)}")

# Function to download file from Azure Blob Storage
def download_from_blob_storage(s3_bucket_name, s3_key, local_file_path):
    """Download a file from S3, or return False if not found."""
    try:
        with open(local_file_path, "wb") as file:
            s3_client.download_fileobj(s3_bucket_name, s3_key, file)
        return True
    except Exception as e:
        if e.response['Error']['Code'] == '404':
            print(f"File not found in S3: {s3_key}")
            return False
        else:
            print(f"Failed to download {s3_key}: {str(e)}")
            return False

# Function to Generate titan embeddings
def generate_titan_embeddings(text):
    try:
        # Generate embeddings using MPNet
        embedding = mpnet_model.encode(text, normalize_embeddings=True)
        return np.array(embedding)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None  # Return None to handle errors gracefully

def call_llm_api(system_message, user_query):
    # Combine system and user messages
    messages = system_message + user_query

    # Prepare the request payload
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16384,
        "messages": [
            {
                "role": "user",
                "content": messages
            }
        ]
    }

    try:
        # Invoke the model (Claude)
        response = bedrock_runtime.invoke_model(
            modelId=INFERENCE_PROFILE_ARN,  # Use the ARN for your inference profile
            contentType='application/json',
            accept='application/json',
            body=json.dumps(payload)
        )

        # Parse and return the response
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']

    except Exception as e:
        return f"An error occurred: {str(e)}"
    
def call_gpt_api(system_message, user_query):
    url = GPT_ENDPOINT
    headers = {  
        "Content-Type": "application/json",  
        "api-key": GPT_API
    }  
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ]
    payload = {  
        "messages": messages,  
        "temperature": 0.7,  
        "max_tokens": 16384   
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()  
    return response.json()["choices"][0]["message"]["content"]

# Faiss index initialization
dimension = 768  # Embedding dimension for text embeddings v3
faiss_index = faiss.IndexFlatL2(dimension)
metadata_store = []

# Updated `save_index_and_metadata` to upload files to Azure Blob Storage
def save_index_and_metadata():
    # Save files locally
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(METADATA_STORE_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    # Upload files to Blob Storage
    try:
        upload_to_blob_storage(FAISS_INDEX_PATH, s3_bucket_name, os.path.basename(FAISS_INDEX_PATH))
        upload_to_blob_storage(METADATA_STORE_PATH, s3_bucket_name, os.path.basename(METADATA_STORE_PATH))
    except Exception as e:
        print(f"Error uploading index or metadata to Blob Storage: {str(e)}")

# Function to load index and metadata
# Load index and metadata from Azure Blob Storage or initialize new
def load_index_and_metadata():
    global faiss_index, metadata_store

    index_blob_name = os.path.basename(FAISS_INDEX_PATH)
    metadata_blob_name = os.path.basename(METADATA_STORE_PATH)

    # Download files from Blob Storage if available
    index_downloaded = download_from_blob_storage(s3_bucket_name, index_blob_name, FAISS_INDEX_PATH)
    metadata_downloaded = download_from_blob_storage(s3_bucket_name, metadata_blob_name, METADATA_STORE_PATH)

    if index_downloaded and metadata_downloaded:
        # Load FAISS index and metadata store
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_STORE_PATH, "rb") as f:
                metadata_store = pickle.load(f)
            print("Index and metadata loaded from Storage.")
        except Exception as e:
            print(f"Failed to load index or metadata: {str(e)}")
            # Initialize empty index and metadata if loading fails
            faiss_index = faiss.IndexFlatL2(dimension)
            metadata_store = []
    else:
        print("Index or metadata not found in Blob Storage. Initializing new.")
        # Initialize empty index and metadata
        faiss_index = faiss.IndexFlatL2(dimension)
        metadata_store = []

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    pages_text = []
    processing_message_placeholder = st.empty()
    progress_bar = st.progress(0)

    for page_num in range(len(doc)):
        processing_message_placeholder.write(f"Processing page {page_num + 1}/{len(doc)}...")
        temp_image_path = None

        try:
            # Render page as an image
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            temp_image_path = os.path.join(tempfile.gettempdir(), f"page_{page_num}.png")
            pix.save(temp_image_path)

            with open(temp_image_path, "rb") as image_file:
                image_bytes = image_file.read()

            # Use basic OCR (DetectDocumentText) instead of full analysis
            response = textract_client.detect_document_text(
                Document={'Bytes': image_bytes}
            )

            page_content = {"page_num": page_num + 1, "text": ""}
            
            # Extract text directly from OCR results
            text_lines = []
            for block in response.get('Blocks', []):
                if block['BlockType'] == 'LINE' and 'Text' in block:
                    text_lines.append(block['Text'])
            
            page_content["text"] = "\n".join(text_lines)
            pages_text.append((page_num + 1, page_content['text'], page_content['text']))

        except Exception as e:
            st.error(f"Error processing page {page_num + 1}: {str(e)}")
            continue

        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)

        progress_bar.progress((page_num + 1) / len(doc))

    return pages_text

def chunk_text(pages_text, chunk_size=1):
    """Create non-overlapping chunks of text from pages."""
    chunks = []
    total_pages = len(pages_text)

    # Loop to create non-overlapping chunks
    for i in range(0, total_pages, chunk_size):
        chunk_parts = []
        for j in range(chunk_size):
            if i + j < total_pages:  # Ensure we do not exceed total pages
                page_num, text = pages_text[i + j]
                chunk_parts.append(f"The following text is from Page {page_num}:\n``````\n\n")
        
        if chunk_parts:  # Only append non-empty chunks
            chunks.append(''.join(chunk_parts))

    return chunks


def add_pdf_to_index(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        temp_pdf_path = temp_pdf.name
        
        pages_text = extract_text_from_pdf(temp_pdf_path)
        
        total_pages = len(pages_text)
        progress_bar = st.progress(0)  # Start at 0%
        
        # Create a placeholder for processing messages
        processing_message_placeholder = st.empty()

        # page_clf_text = ""
        
        for page_num, text, page_content in pages_text:
            processing_message_placeholder.write(f"Processing page {page_num}/{total_pages}...")

            embedding = generate_titan_embeddings(text)
            faiss_index.add(embedding.reshape(1, -1))
            metadata_store.append({
                "filename": os.path.basename(pdf_file.name),
                "page": page_num,
                "text": page_content
            })
            
            # Update progress bar and clear previous message after updating
            progress_bar.progress(page_num / total_pages)  # Update to reflect current page
        
        save_index_and_metadata()
        
        try:
            # Upload the processed file to Azure Blob Storage
            blob_name = os.path.basename(pdf_file.name)
            upload_to_blob_storage(temp_pdf_path, s3_bucket_name, blob_name)
            
            os.remove(temp_pdf_path)  # Ensure file is removed after processing
        except PermissionError:
            st.warning("Could not delete the temporary PDF file. It might still be in use.")

def get_page_range_for_files(selected_files):
    """
    Finds the min and max page numbers for the selected files from metadata.
    """
    page_ranges = {}
    
    for file in selected_files:
        # Filter metadata for the selected file
        file_metadata = [metadata for metadata in metadata_store if metadata['filename'] == file]
        
        # Get the minimum and maximum page numbers for the file
        if file_metadata:
            min_page = min(metadata['page'] for metadata in file_metadata)
            max_page = max(metadata['page'] for metadata in file_metadata)
            page_ranges[file] = (min_page, max_page)
    
    return page_ranges

def query_documents_with_page_range(selected_files, selected_page_ranges, prompt, top_k, last_messages, web_search, llm_model):
    # print(selected_files)
    # print(selected_page_ranges)
    qp_prompt = {
        "system_message": "You are an intelligent query refiner. Your job is to take a user's original query (which may contain poor grammar or informal language) along with the last 5 messages of the conversation and generate two well-formed prompts: one for a semantic search over a FAISS index and another for a web search. The semantic search prompt should improve the user's provided query, incorporating the last 5 messages for better contextual understanding. The web search prompt should refine the query further to fetch relevant legal resources online. Output only a JSON object with 'semantic_search_prompt' and 'web_search_prompt' as keys.",
        "user_query": f"User Query: {prompt}\n\nLast 5 Messages: {last_messages}\n\nGenerate the JSON output with the two improved prompts."
    }

    op_format = '''
    # Output Format:
    
    ```json
    {
        "semantic_search_prompt": "Refined user query using proper grammar and context from last 5 messages, optimized for FAISS index retrieval.",
        "web_search_prompt": "Further refined query designed to fetch relevant legal resources from the web."
    }
    ```
    '''

    prompts = call_llm_api(qp_prompt["system_message"], qp_prompt["user_query"]+op_format)
    try:
        # return json.loads(answer[7:-3])
        prompt_op = json.loads(prompts.split("```json")[1].split("```")[0])
    except:
        # return json.loads(answer[3:-3])
        prompt_op = json.loads(prompts.split("```")[1].split("```")[0])
    print(prompt_op)
    query = prompt_op["semantic_search_prompt"]

    query_embedding = generate_titan_embeddings(query).reshape(1, -1)
    if faiss_index.ntotal == 0:
        st.error("The FAISS index is empty. Please upload a PDF to populate the index.")
        return [], "No data available to query."

    # Fetch all metadata for the given query
    k = faiss_index.ntotal  # Initial broad search
    distances, indices = faiss_index.search(query_embedding, k)
    
    filtered_results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata_store):
            metadata = metadata_store[idx]
            if metadata['filename'] in selected_files:
                min_page, max_page = selected_page_ranges.get(metadata['filename'], (None, None))
                if min_page and max_page and min_page <= metadata['page'] <= max_page:
                    filtered_results.append((dist, idx))
    
    # Limit to topK after filtering
    top_k_results = sorted(filtered_results, key=lambda x: x[0])[:top_k]
    top_k_metadata = [metadata_store[idx] for _, idx in top_k_results]

    user_query = f"""
    You are required to provide a structured response to the following question, based on the context retrieved from the provided documents.

    # User Query:
    <<<{prompt}>>>

    # The top K most relevant contexts fetched from the documents are as follows:
    {json.dumps(top_k_metadata, indent=4)}

    # The last few messages of the conversation to help you maintain continuity and relevance:
    {json.dumps(last_messages)}

    # Always Approach the Task Step by Step.
    * Read and Understand the Provided Contexts.
    * Identify Relevant sections from those and Arrange them as necessary
    * Then Formulate your Answer Adhering to the Guidelines.
    """

    ws_response = ""

    if web_search:
        ws_query = prompt_op["web_search_prompt"]
        # Call the LLM API to get the answer
        # To install, run: pip install tavily-python


        client = TavilyClient(api_key=TAVILY_API)

        ws_response = client.search(
            query=ws_query,
            search_depth="advanced",
            include_raw_content=True
        )

        print(ws_response)

        wsp = f"""
        # Feel free to use the Web Search Results for Additional Context as well:

        {json.dumps(ws_response)}
        """
        if llm_model=="Claude 3.5 Sonnet":
            answer = call_llm_api(system_message, user_query+wsp)
        elif llm_model=="GPT 4o":
            answer = call_gpt_api(system_message, user_query+wsp)
    else:
        if llm_model=="Claude 3.5 Sonnet":
            answer = call_llm_api(system_message, user_query)
        elif llm_model=="GPT 4o":
            answer = call_gpt_api(system_message, user_query)

    return top_k_metadata, answer, ws_response

def final_format(top_k_metadata, answer, ws_response):
    sys_msg = """
    You are a helpful Legal Assistant. 
    You Specialise in Formatting Generated Answers.

    """
    
    input_context = f"""
    Below is the Generated Answer Fetched from LLM:
    <<<{answer}>>>

    The top K most relevant contexts fetched from the documents are as follows:
        {json.dumps(top_k_metadata, indent=4)}

    ##########################################################################
        
    The Web Search Results Fetched are as follows:
        {json.dumps(ws_response)}

    ##########################################################################
        
    """

    final_op_format = '''
        For Transparency and Explanability, we Would like you to do the following:
            1. Break the Answer into Logical Shards.
            2. For Each Shard, Map them to the Relevant Sources (From the Provided TopK Context) in the Formatting as Presented below
        
        Note: The Shards when concatenated should help us regenerate the provided `Generated Answer`
        Approach the Above Step by Step.

        # Final Output Format:
        ```
        {
            "segmented_answer":    
                [
                    {
                        "section": "The first shard of the generated answer",
                        "sources": [
                                        {"filename": "The First Filename", "page": "page_num", "text": "The Relevant text from the page only"},
                                        {"filename": "The Second Filename", "page": "page_num", "text": "The Relevant text from the page only"},
                                        . # Add more sources if necessary
                                    ]
                    },
                    {
                        "section": "The second part of the generated answer", 
                        "sources": [
                                        {"filename": "The First Filename", "page": "page_num", "text": "The Relevant text from the page only"},
                                        {"filename": "The Second Filename", "page": "page_num", "text": "The Relevant text from the page only"},
                                        . # Add more sources if necessary
                                    ]
                    },
                    .
                    .
                ]
        }
        ```
    '''
    # Call the LLM API to get the answer
    answer = call_llm_api(sys_msg, input_context+final_op_format)
    try:
        # return json.loads(answer[7:-3])
        return json.loads(answer.split("```json")[1].split("```")[0])
    except:
        # return json.loads(answer[3:-3])
        return json.loads(answer.split("```")[1].split("```")[0])

USERS = load_dict_from_json(users_file)

def login():
    display_logo()
    st.title("Ready to Dive In? Sign In!")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username in USERS and password == USERS[username]:
            st.session_state.authenticated = True
            st.session_state.username = username  # Store the logged-in user's name
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

def logout():
    """Logout function to clear authentication."""
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun(),

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        login()
        return

    display_logo()
    st.title("Document Query Assistant")

    # Initialize session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = None
    
    if 'sources' not in st.session_state:
        st.session_state.sources = []

    # if 'chat_history' not in st.session_state:
    #     st.session_state.chat_history = []
    if 'chat_history' not in st.session_state:
            st.session_state.chat_history = load_chat_history()

    # Retrieve the current user’s conversation history:
    current_user = st.session_state.username
    conversations = st.session_state.chat_history.get(current_user, [])

    # Call `load_index_and_metadata` during app initialization
    load_index_and_metadata()
    st.sidebar.header(f"Hello {current_user}")

    logout()

    st.sidebar.header("Options")

    option = st.sidebar.selectbox("Choose an option", ["Upload PDF", "Query Documents"])

    if option == "Upload PDF":
        # ... (existing PDF upload code remains the same)
        st.header("Upload PDF Documents")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.write(f"Processing {uploaded_file.name}...")
                if file_exists_in_blob(uploaded_file.name):
                    st.warning(f"File '{uploaded_file.name}' already exists in Blob Storage. Skipping upload.")
                else:
                    add_pdf_to_index(uploaded_file)
                    st.success(f"File '{uploaded_file.name}' has been successfully uploaded and added to the index.")

    elif option == "Query Documents":
        st.header("Query Documents")
        st.sidebar.header("Settings")

        llm_model = st.sidebar.selectbox("Choose Your Model", ["Claude 3.5 Sonnet", "GPT 4o"])

        # Add "New Chat" button
        if st.sidebar.button("New Chat"):
            # Reset current conversation and messages
            st.session_state.current_conversation = None
            st.session_state.messages = []
            st.session_state.sources = []
            st.success("Started a new conversation.")

        
        web_search = st.sidebar.toggle("Enable Web Search")

        # Sidebar slider for Top-K selection
        top_k = st.sidebar.slider(
            "Select Top-K Results",  # Label
            min_value=1,             # Minimum value
            max_value=50,            # Maximum value
            value=20,                # Default value
            step=1                   # Step size
        )

        

        # File and Page Range Selection (similar to existing code)
        available_files = list(set([metadata['filename'] for metadata in metadata_store]))

        if available_files:
            # File selection
            selected_files = st.multiselect(
                "Select up to 4 files to include in the query:",
                available_files
            )

            if len(selected_files) > 4:
                st.warning("You can select a maximum of 4 files.")
                return

            # Page range selection
            page_ranges = get_page_range_for_files(selected_files)
            selected_page_ranges = {}

            for file in selected_files:
                min_page, max_page = page_ranges[file]
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    start_page = st.number_input(
                        f"Start page for {file}",
                        min_value=min_page,
                        max_value=max_page,
                        value=min_page
                    )
                with col2:
                    end_page = st.number_input(
                        f"End page for {file}",
                        min_value=min_page,
                        max_value=max_page,
                        value=max_page
                    )

                selected_page_ranges[file] = (start_page, end_page)

        # Sidebar Header
        st.sidebar.header("Previous Conversations")

        # Load conversations
        # conversations = list(load_conversations_from_blob())
        # In your main() function, after loading the chat history:
        current_user = st.session_state.username
        user_conversations = st.session_state.chat_history.get(current_user, [])

        # Ensure unique conversations while preserving order
        seen_labels = {}
        unique_conversations = []

        # for conv in conversations:
        #     conv_label = conv.get('messages', [])[0]["content"]
            
        #     if conv_label not in seen_labels:
        #         seen_labels[conv_label] = conv  # Store the conversation
        #         unique_conversations.append(conv)

        for conv in user_conversations:
            # Use first message content as a preview label (adjust as needed)
            conv_label = conv.get('messages', [])[0]["content"]
            if conv_label not in seen_labels:
                seen_labels[conv_label] = conv
                unique_conversations.append(conv)

        # # Sort by timestamp (if available), latest first
        # unique_conversations.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        # Sort by timestamp (latest first)
        unique_conversations.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        # Display in the sidebar
        for idx, conv in enumerate(unique_conversations):
            conv_label = conv.get('messages', [])[0]["content"][:20]
            if 'timestamp' in conv:
                conv_label += f" ({conv['timestamp']})"
            if st.sidebar.button(conv_label, key=f"conv_{idx}"):
                st.session_state.current_conversation = conv
                st.session_state.messages = conv.get('messages', [])
                
                # Restore file and page range selections
                selected_files = conv.get('files', [])
                selected_page_ranges = conv.get('page_ranges', {})
                
                # Reapply file and page range selections
                st.multiselect(
                    "Selected Files", 
                    options=available_files, 
                    default=selected_files
                )
                
                # Optionally, you can programmatically set the page ranges
                for file, (start, end) in selected_page_ranges.items():
                    st.number_input(
                        f"Start page for {file}", 
                        value=start, 
                        key=f"start_{file}"
                    )
                    st.number_input(
                        f"End page for {file}", 
                        value=end, 
                        key=f"end_{file}"
                    )

        # Display Current Conversation
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Task Selection
        task = st.selectbox(
            "Select a task:",
            options=list(prompt_library.keys()),
            index=0
        )

        # Chat Input
        if prompt := st.chat_input("Enter your query"):
            # Add user message to chat history
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt
            })

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            if len(st.session_state.messages) >= 5:
                last_messages = st.session_state.messages[-5:]
            else:
                last_messages = st.session_state.messages

            # Process query
            with st.spinner("Searching documents..."):
                # Use the selected files and page ranges
                top_k_metadata, answer, ws_response = query_documents_with_page_range(
                    selected_files, 
                    selected_page_ranges, 
                    prompt,
                    top_k,
                    last_messages,
                    web_search,
                    llm_model
                )
                
                st.session_state.sources.append({
                    "top_k_metadata": top_k_metadata,
                    "answer": answer,
                    "websearch_metadata": ws_response
                })
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer
                })

                # Create a one-line version for display (removing newline characters)
                display_answer = " ".join(answer.splitlines())

                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                # Add an expander for a copy-friendly version
                with st.expander("Show Copyable Answer"):
                    # st.code(answer.replace("\n", "\\n"), language="text")  # language can be set to "" if you want plain text formatting
                    st.code(answer, language="text")

                ist_timezone = pytz.timezone('Asia/Kolkata')
                new_conversation = {
                    "timestamp": datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M:%S"),
                    "messages": st.session_state.messages,
                    "files": selected_files,
                    "page_ranges": selected_page_ranges
                }

                # Update the chat history dictionary for the current user:
                user = st.session_state.username
                if user in st.session_state.chat_history:
                    st.session_state.chat_history[user].append(new_conversation)
                else:
                    st.session_state.chat_history[user] = [new_conversation]

                # Optional: Limit to a maximum number of conversations per user (e.g., last 10)
                if len(st.session_state.chat_history[user]) > 10:
                    st.session_state.chat_history[user] = st.session_state.chat_history[user][-10:]

                save_chat_history(st.session_state.chat_history)

        # Source Mapping Button (Optional)
        if st.button("Show Source Mapping"):
            if len(st.session_state.sources)>0:
                top_k_metadata = st.session_state.sources[-1]["top_k_metadata"]
                answer = st.session_state.sources[-1]["answer"]
                ws_query = st.session_state.sources[-1]["websearch_metadata"]
                with st.spinner("Mapping Source..."):
                    final_answer = final_format(top_k_metadata, answer, ws_query)
                    st.write(final_answer)

                    # Relevant pages
                    st.subheader("Relevant Pages")
                    for metadata in top_k_metadata:
                        with st.expander(f"Filename: {metadata['filename']}, Page: {metadata['page']}"):
                            st.write(f"```{metadata['text']}```")
            else:
                st.warning("No recent query to map sources for.")

    else:
        st.warning("No files available in the index. Please upload PDFs to populate the index.")

if __name__ == "__main__":
    main()
