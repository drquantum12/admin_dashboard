from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from preprocessing.preprocess import preprocess_router
from settings.config import settings_router
from research_agent.agent import agent_router
from preprocessing.quiz_extraction import preprocess_quiz_router
import os


    
app = FastAPI()

app.include_router(preprocess_router, prefix="/preprocess", tags=["preprocess"])
app.include_router(settings_router, prefix="/settings", tags=["settings"])
app.include_router(agent_router, prefix="/research", tags=["research"])
app.include_router(preprocess_quiz_router, prefix="/preprocess/quiz", tags=["preprocess-quiz"])

static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, status_code=200)


