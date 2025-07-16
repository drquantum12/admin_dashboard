from io import BytesIO
import pymupdf as fitz
import os
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from bson import ObjectId
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
# from langchain.chains import LLMChain
from prompt_templates import content_extraction_prompt
from pydantic import BaseModel
from google.cloud import storage
from utility.mongo_client import db

class PDFRequest(BaseModel):
    pdf_path: str
    prompt_id: str

class UploadRequest(BaseModel):
    pdf_path: str
    markdown: str

BUCKET_NAME = os.getenv("BUCKET_NAME")
RAW_DATA_PREFIX = os.getenv("RAW_DATA_PREFIX", "raw-data/")
PROCESSED_DATA_PREFIX = os.getenv("MARKDOWN_PROCESSED_DATA_PREFIX", "processed-data/markdowns/")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

storage_client = storage.Client()
prompt_collection = db["prompts"]

preprocess_router = APIRouter()


llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=1,
            max_output_tokens=8192,
            timeout=30,
            max_retries=2,)
# llm = ChatOllama(model="llama3.2:latest", temperature=0.1)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    return "\n".join(page.get_text() for page in reader)

@preprocess_router.get("/", response_class=HTMLResponse)
async def home():
    html_path = os.path.join(os.path.dirname(__file__), "..", "static", "preprocess.html")
    with open(html_path, encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, status_code=200)

@preprocess_router.get("/list-files")
def list_files():
    bucket = storage_client.bucket(BUCKET_NAME)

    # Get all PDF and MD files
    all_pdfs = [
        blob.name for blob in bucket.list_blobs(prefix="raw-data/")
        if blob.name.endswith(".pdf")
    ]
    all_mds = [
        blob.name for blob in bucket.list_blobs(prefix="processed-data/markdowns/")
        if blob.name.endswith(".md")
    ]

    # Normalize file paths to get relative logical keys
    def normalize(path: str, prefix: str, extension: str) -> str:
        return path.replace(prefix, "").replace(extension, "")

    normalized_pdfs = set(normalize(p, "raw-data/", ".pdf") for p in all_pdfs)
    normalized_mds = set(normalize(m, "processed-data/markdowns/", ".md") for m in all_mds)

    # Filter only those PDFs whose MD does not exist
    unprocessed_pdfs = [
        "raw-data/" + path + ".pdf"
        for path in normalized_pdfs - normalized_mds
    ]

    unprocessed_pdfs = [
        "raw-data/" + path + ".pdf" for path in normalized_pdfs
        if path not in normalized_mds
    ]

    return {
        "unprocessed_pdfs": unprocessed_pdfs,
        "processed_mds": all_mds  # Send full paths for tree rendering
    }

@preprocess_router.post("/process-pdf-stream")
async def process_pdf_stream(req: PDFRequest):
    prompt_id = req.prompt_id
    blob = storage_client.bucket(BUCKET_NAME).blob(req.pdf_path)
    pdf_data = blob.download_as_bytes()
    raw_text = extract_text_from_pdf(pdf_data)

    # Get prompt content from DB
    prompt_data = prompt_collection.find_one({"_id": ObjectId(prompt_id)})
    if not prompt_data:
        return JSONResponse(status_code=404, content={"error": "Prompt not found"})

    full_prompt = prompt_data["prompt"]  # Assuming 'prompt' holds the string prompt

    def gen():
        prompt = [{"role": "system", "content": full_prompt},
                  {"role": "user", "content": f"Document:\n{raw_text}"}]
        for chunk in llm.stream(prompt):
            yield chunk.content

    return StreamingResponse(gen(), media_type="text/plain")


@preprocess_router.post("/upload-md")
def upload_md(req: UploadRequest):
    output_path = req.pdf_path.replace("raw-data/", "processed-data/markdowns/").replace(".pdf", ".md")
    blob = storage_client.bucket(BUCKET_NAME).blob(output_path)
    blob.upload_from_string(req.markdown, content_type="text/markdown")
    return {"message": f"✅ Markdown uploaded to: {output_path}"}


@preprocess_router.get("/list-prompts")
def list_prompts():
    prompts = prompt_collection.find({}, {"_id": 1, "subject": 1})
    return [{"id": str(p["_id"]), "subject": p["subject"]} for p in prompts]



def remove_unwanted_markdown_lists(text: str) -> str:
    lines = text.splitlines()
    clean_lines = [
        line for line in lines
        if not line.strip().startswith(("* ", "- ", "1.", "2.", "3.")) and not line.strip().startswith("*\t")
    ]
    return "\n".join(clean_lines)

