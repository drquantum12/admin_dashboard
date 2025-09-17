from pymongo import MongoClient
from pydantic import BaseModel
import os
import fitz  # PyMuPDF
from google.cloud import storage
from google.cloud import firestore


BUCKET_NAME = os.getenv("BUCKET_NAME")
RAW_DATA_PREFIX = os.getenv("RAW_DATA_PREFIX", "raw-data/")
PROCESSED_DATA_PREFIX = os.getenv("MARKDOWN_PROCESSED_DATA_PREFIX", "processed-data/markdowns/")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

storage_client = storage.Client()
firestore_client = firestore.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def get_board_grade_subject_chapter():
    # example format "processed-data/vectordb_chunks/CBSE/GRADE_9/krutika/Chapter_3_Reed ki haddi.json"
    data = []
    blobs = [
        blob for blob in bucket.list_blobs(prefix="processed-data/vectordb_chunks/")
        if blob.name.endswith(".json")
    ]

    for blob in blobs:
        parts = blob.name.split('/')
        if len(parts) >= 5:
            board = parts[2]
            grade = parts[3]
            subject = parts[4]
            chapter = parts[5].replace('.json', '') if len(parts) > 5 else None
            data.append({
                "board": board,
                "grade": grade,
                "subject": subject,
                "chapter": chapter
            })
    return data

if __name__ == "__main__":
    data = get_board_grade_subject_chapter()
    firestore_client.collection("available-academic-data").document("metadata").set({"data": data}, merge=True)
    print("data uploaded to firestore")

    # chunks = split_pdf_text(pdf_text)
    # print(len(chunks), "chunks created from PDF text.")