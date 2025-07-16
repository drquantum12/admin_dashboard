from pymongo import MongoClient
from pydantic import BaseModel
import os
import fitz  # PyMuPDF


client = MongoClient(os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017"))
db = client["neurosattva"]
prompt_collection = db["prompts"]

class Prompt(BaseModel):
    subject: str
    prompt: str

pdf_text = ""
doc = fitz.open("data/niti_aayog_ed.pdf")
for page in doc:
    pdf_text += page.get_text()

if __name__ == "__main__":
    pass
    # chunks = split_pdf_text(pdf_text)
    # print(len(chunks), "chunks created from PDF text.")