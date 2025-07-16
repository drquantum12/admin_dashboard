from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import JSONResponse, HTMLResponse
from pymongo import MongoClient
from pydantic import BaseModel
import os
from utility.mongo_client import db

app = FastAPI()

collection = db["prompts"]

class PromptUpdate(BaseModel):
    subject: str
    prompt: str

settings_router = APIRouter()

@settings_router.get("/", response_class=HTMLResponse)
async def home():
    html_path = os.path.join(os.path.dirname(__file__), ".." , "static", "settings.html")
    with open(html_path, encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, status_code=200)

@settings_router.get("/prompts")
async def get_prompts():
    prompts = {}
    for doc in collection.find():
        prompts[doc["subject"]] = doc["prompt"]
    return JSONResponse(content={"prompts": prompts})

@settings_router.post("/prompts/update")
async def update_prompt(data: PromptUpdate):
    collection.update_one(
        {"subject": data.subject},
        {"$set": {"prompt": data.prompt}},
        upsert=True
    )
    return {"message": "Prompt updated successfully"}