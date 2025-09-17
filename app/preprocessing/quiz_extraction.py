from typing import TypedDict, List
from typing import Any
from langgraph.graph import StateGraph
from fastapi import APIRouter, UploadFile, File, Request, Body
from utility.mongo_client import db
from uuid import uuid4
from datetime import datetime
import hashlib
import fitz  # PyMuPDF
from PIL import Image
import io, json, tempfile, os
import google.generativeai as genai
from google.cloud import storage
from fastapi.responses import HTMLResponse, StreamingResponse

quiz_collection = db["quizzes"]
storage_client = storage.Client()
preprocess_quiz_router = APIRouter()

@preprocess_quiz_router.get("/", response_class=HTMLResponse)
async def extract_quiz_home():
	html_path = os.path.join(os.path.dirname(__file__), "../static", "extract-quiz.html")
	with open(html_path, encoding="utf-8") as f:
		content = f.read()
	return HTMLResponse(content=content, status_code=200)

@preprocess_quiz_router.post("/upload-to-db")
async def upload_quiz_to_db(quizzes: list = Body(...)):
	# Insert quizzes into MongoDB collection
	if not quizzes or not isinstance(quizzes, list):
		return {"success": False, "message": "No quiz data provided."}
	try:
		quizzes = [{**quiz, "_id": str(uuid4()), "created_at": datetime.now()} for quiz in quizzes]
		result = quiz_collection.insert_many(quizzes)
		return {"success": True, "inserted_ids": [str(_id) for _id in result.inserted_ids]}
		
	except Exception as e:
		return {"success": False, "message": str(e)}

# Endpoint to process uploaded PDF and extract quiz
@preprocess_quiz_router.post("/extract-quiz")
async def extract_quiz_endpoint(request: Request, pdf: UploadFile = File(...)):
	# Save uploaded PDF to a temp file
	with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
		tmp.write(await pdf.read())
		tmp_path = tmp.name

	# Get Google API key from environment or config (customize as needed)
	google_api_key = os.getenv("GOOGLE_API_KEY")

	def stream_quiz_extraction():
		images = extract_pdf_page_images(tmp_path)
		genai.configure(api_key=google_api_key)
		quiz_questions = []
		for page_number, image in enumerate(images):
			yield f"Processing page {page_number + 1}...\n"
			try:
				model = genai.GenerativeModel('models/gemini-2.5-flash')
				prompt = (
    "You are an expert at extracting questions from test papers. "
    "Analyze the following image of a test paper page and extract all multiple-choice questions "
    "and their answer options. Return the results as a list of JSON objects. "
    "The JSON object for each question should have: "
    " - 'question': the question text "
    " - 'options': an array of strings for the answer choices "
    " - 'correct_answer': an array of strings with the correct option(s), if present. "
    "\n\nLanguage rules: "
    " - For language subjects (e.g., Hindi, Sanskrit, French, etc.), keep the questions in the original subject language. "
    " - For all other subjects (e.g., Mathematics, Science, Physics, Chemistry, Biology, etc.), always extract questions in English. "
    " - If the same question appears in multiple languages, keep only one version (following the above rules). "
    "\n\nFormatting rules: "
    " - Always extract any mathematical formulae, equations, or expressions in proper LaTeX format so they can be rendered later. "
    " - Do not include any text outside the JSON block."
    "\n\nExample Output:\n"
    '[{"question": "What is the capital of France?", '
    '"options": ["London", "Paris", "Berlin", "Delhi"], '
    '"correct_answer": ["Paris"]}, '
    '{"question": "Solve for x: $2x + 5 = 15$", '
    '"options": ["$x = 5$", "$x = 10$", "$x = 2$", "$x = 15$"], '
    '"correct_answer": ["$x = 5$"]}]'
)
				response = model.generate_content([prompt, image])
				response.resolve()
				json_str = response.text.strip().replace("```json", "").replace("```", "")
				extracted_data = json.loads(json_str)
				print(f"✅ Extracted {len(extracted_data)} questions from page {page_number + 1}")
			except Exception as e:
				yield f"Error during Gemini API call for page {page_number + 1}: {e}\n"
				print(f"❌ Error during Gemini API call for page {page_number + 1}: {e}")
				extracted_data = []
			quiz_questions.extend(extracted_data)
			# print(f"quiz questions : {quiz_questions}")
		yield json.dumps(quiz_questions)

	return StreamingResponse(stream_quiz_extraction(), media_type="text/plain")

# endpoint to upload image on cloud storage and return the public url
@preprocess_quiz_router.post("/upload-image")
async def upload_image(image: UploadFile = File(...)):
    # Read file content
    content = await image.read()

    # Compute SHA256 hash of the image
    file_hash = hashlib.sha256(content).hexdigest()

    bucket = storage_client.bucket("question-image-v1")
    blob = bucket.blob(f"question-image/{file_hash}.png")

    # If file already exists, return its public URL
    if blob.exists():
        return {"success": True, "url": blob.public_url, "duplicate": True}

    # Otherwise, save temporarily and upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    blob.upload_from_filename(tmp_path, content_type="image/png")

    return {"success": True, "url": blob.public_url, "duplicate": False}

# Function to extract page images from PDF
def extract_pdf_page_images(pdf_path: str) -> List[Image.Image]:
	"""
	Given a PDF file path, returns a list of PIL Image objects for each page.
	"""
	images = []
	try:
		doc = fitz.open(pdf_path)
		for page_num in range(len(doc)):
			page = doc.load_page(page_num)
			pix = page.get_pixmap(dpi=300)  # Render at 300 DPI
			img_bytes = pix.tobytes("png")
			image = Image.open(io.BytesIO(img_bytes))
			images.append(image)
		doc.close()
	except Exception as e:
		print(f"Error extracting images from PDF: {e}")
	return images

# LangGraph agent code for extracting quiz questions using Gemini VLM
class QuizQuestion(TypedDict):
	question: str
	options: List[str]
	correct_answer: List[str]

def extract_quiz_from_pdf(pdf_path: str, gemini_api_key: str) -> List[QuizQuestion]:
	images = extract_pdf_page_images(pdf_path)
	genai.configure(api_key=gemini_api_key)
	quiz_questions = []
	for page_number, image in enumerate(images):
		try:
			model = genai.GenerativeModel('models/gemini-2.5-flash')
			prompt = (
				"You are an expert at extracting questions from test papers. "
				"Analyze the following image of a test paper page and extract all multiple-choice questions "
				"and their answer options. Return the results as a list of JSON objects. "
				"The JSON object for each question should have a 'question' key, 'options' and 'correct_answer' (if present) "
				"key, which is an array of strings. Do not include any text outside the JSON block."
				"\n\nExample Output:\n"
				'[{"question": "What is the capital of France?", "options": ["London", "Paris", "Berlin", "Delhi"], "correct_answer": ["Paris"]}]'
			)
			buf = io.BytesIO()
			image.save(buf, format='PNG')
			buf.seek(0)
			image_bytes = buf.read()
			response = model.generate_content([prompt, image_bytes])
			response.resolve()
			# Clean up the response to get a valid JSON string
			json_str = response.text.strip().replace("```json", "").replace("```", "")
			extracted_data = json.loads(json_str)
		except Exception as e:
			print(f"❌ Error during Gemini API call for page {page_number + 1}: {e}")
			extracted_data = [] # Return an empty list if there's an error
		quiz_questions.extend(extracted_data)
	return quiz_questions