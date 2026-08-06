import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Initialize the client using Environment Variables
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in environment variables. Please check your .env file.")

client = genai.Client(api_key=api_key)

# Initialize FastAPI App
app = FastAPI(
    title="Ma3refa AI Quiz Engine",
    description="API for generating adaptive educational quizzes using Gemini AI",
    version="1.0.0"
)

# Allow CORS for Flutter/Frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your app's specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the Data Model for incoming requests (Data Flow Architecture)
class QuizRequest(BaseModel):
    category: str
    sub_category: str
    difficulty: str
    age_group: str
    language: str = "Arabic"
    num_questions: int = 10

# The Core AI Engine Endpoint
@app.post("/api/generate-quiz")
async def generate_quiz_endpoint(request: QuizRequest):
    """
    Endpoint to generate Multiple Choice Questions (MCQs) using Google Gemini AI.
    Accepts JSON body with quiz parameters and returns generated JSON quiz.
    """
    system_prompt = f"""
    You are an academic professor and a strict scientific reviewer. Your goal is to evaluate and increase the user's knowledge through Multiple Choice Questions (MCQs).

    Context:
    - Main Category: {request.category}
    - Sub-Category: {request.sub_category}
    - Difficulty Level: {request.difficulty}
    - Target Age Group: {request.age_group}
    - Number of Questions: {request.num_questions}
    - Output Language: {request.language}

    Content Drafting Instructions:
    1. Formulate the language of the questions, options, and the "explanation" style to perfectly suit the cognitive and linguistic awareness of the target age group ({request.age_group}).
    2. The questions must be 100% scientifically accurate. Avoid fabricating undocumented information.
    3. Do not repeat questions. Ensure each question covers a different concept within the sub-category.
    4. ⚠️ VERY IMPORTANT: Never use double quotes (") inside the texts of the questions, explanations, or options. Use single quotes (') instead to avoid breaking the JSON structure.
    5. ⚠️ TECHNICAL TERMS RULE: Write the content in the specified Output Language ({request.language}). HOWEVER, you MUST keep all technical terms, programming keywords, data types, and scientific concepts in their original English form. DO NOT translate technical terms into Arabic (e.g., write "نستخدم الـ List" instead of "نستخدم القائمة", and use "float" instead of "عدد عشري").
    6. The response MUST be in strictly valid JSON format only, without any markdown formatting or additional text, matching this exact structure:
    {{
      "quiz_title": "Quiz title in {request.language} (Keep technical terms in English)",
      "questions": [
        {{
          "question": "Question text here in {request.language} (Technical terms in English)",
          "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
          "correct_index": 0,
          "explanation": "Detailed scientific explanation tailored for the age group in {request.language} (Technical terms in English)"
        }}
      ]
    }}
    """

    try:
        # Call the API using the selected fast model
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=system_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2, # Low temperature for more factual output
            )
        )

        # Clean the received text to ensure successful JSON parsing
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        # Parse the string into a Python Dictionary
        quiz_data = json.loads(raw_text.strip())
        return quiz_data

    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse data from AI as valid JSON.")
    except Exception as e:
        print(f"API Connection Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

