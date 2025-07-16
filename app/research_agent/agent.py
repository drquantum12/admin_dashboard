from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .research_main import build_research_agent, initial_state
import pymupdf as fitz
import asyncio, re, os

agent_router = APIRouter()

def extract_chunks_from_pdf(file_bytes: bytes) :
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    raw_text = "\n".join(page.get_text() for page in doc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=40960, chunk_overlap=256)
    chunks = [chunk.page_content for chunk in splitter.create_documents([raw_text])]
    return chunks

@agent_router.get("/", response_class=HTMLResponse)
async def home():
    html_path = os.path.join(os.path.dirname(__file__), "..", "static", "research-agent.html")
    with open(html_path, encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, status_code=200)

@agent_router.post("/process-research")
async def process_research(
    file: UploadFile,
    objective: str = Form(...),
):
    file_bytes = await file.read()
    chunks = extract_chunks_from_pdf(file_bytes)
    state = initial_state(objective=objective, chunks=chunks)
    agent = build_research_agent()

    async def generate():
        last_node = None
        yield f"\n🧠 Objective: {objective}\n"

        for chunk, metadata in agent.stream(state, stream_mode="messages"):
            current_node = metadata.get("langgraph_node")
            state_data = metadata.get("state", {})

            if current_node != last_node:
                if current_node == "DEFINE":
                    yield f"\n##🔍 Objective Definition:\n"
                    content = state_data.get("objective_definition", "")
                elif current_node == "PLAN":
                    yield f"\n##📝 Research Plan:\n"
                    content = state_data.get("plan", "")
                elif current_node == "GATHER":
                    idx = state_data.get("current_chunk_index", 0) + 1
                    total = len(chunks)
                    gathered = state_data.get("gathered", [""])
                    yield f"\n##📚 Insights from Chunk {idx}/{total}:\n"
                    content = gathered[-1]
                elif current_node == "REFINE":
                    yield f"\n##🔧 Refined Research Notes:\n"
                    content = state_data.get("refined", "")
                elif current_node == "GENERATE":
                    yield f"\n##📄 Final Report:\n"
                    content = state_data.get("final_output", "")
                else:
                    content = chunk.content
            else:
                content = chunk.content

            # Simulate streaming chunk by chunk
            tokens = re.split(r'(\s+)', content)  # splits but keeps spaces & newlines

            for token in tokens:
                yield token
                await asyncio.sleep(0.03)  # Simulate a small delay for streaming effect

            last_node = current_node

    return StreamingResponse(generate(), media_type="text/plain")

@agent_router.post("/save-research")
async def save_research(
    file: UploadFile,
    response_text: str = Form(...),
):
    filename = file.filename.replace(".pdf", "_research.txt")
    with open(f"./research_outputs/{filename}", "w") as f:
        f.write(response_text)

    return JSONResponse(content={"message": f"Research saved successfully as {filename}."}, status_code=200)

@agent_router.post("get-pdf-pages")
async def get_pdf_pages(file: UploadFile):
    file_bytes = await file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return JSONResponse(content={"page_count": len(doc)}, status_code=200)
