# Ma3refa AI Engine

>This repository contains a FastAPI microservice that generates quizzes using Gemini API.

## Libraries Used



* **`fastapi`**: The core web framework for building the RESTful API endpoints.


* **`uvicorn`**: The ASGI web server implementation used to run the FastAPI application.


* **`pydantic`**: For validation and defining the schema in request.


* **`google-genai`**: Used to initialize the client and communicate with the model.


* **`python-dotenv`**: Used to load the `GEMINI_API_KEY` and `SECRET_KEY` securely from the environment_variables.



## Project Files Structure

```text
.
|── app.py              
|── requirements.txt     
└── README.md            

```

## App Flow 

1. **Request Handling**: The main backend service sends an HTTP POST request to the `/api/generate-quiz` endpoint with the quiz configuration payload.


2. **Authentication**: The `verify_internal_key` dependency intercepts the request to check the `x-internal-key` header. If it does not match the server's `SECRET_KEY`, the request is immediately rejected with a 401 Unauthorized status.


3. **Data Validation**: The incoming JSON is validated against the `QuizRequest` model, ensuring parameters like `num_questions` (constrained between 5 and 20) and `language` are valid.


4. **Prompt Construction**: The application compiles the `system_instruction` and the `user_content`.


5. **AI Generation**: The compiled prompts are sent to the `gemini-3.5-flash-lite` model via the Google GenAI client, with the response schema enforced natively to ensure structured JSON output.


6. **Response Parsing**: The API strips any residual markdown formatting (like ````json`) from the Gemini response, parses it, and returns the clean JSON object back to the backend.

## How to Install

1. **Clone the repository** to your local machine.
```bash
git clone https://github.com/Ma3refa-GP-IEEE/ma3refa-ai.git
cd ma3refa-ai

```

2. **Install dependencies** from the requirements file:


```bash
pip install -r requirements.txt

```


3. **Set up Environment Variables**: Create a `.env` file in the root directory and define the following variables:


```env
GEMINI_API_KEY=your_google_gemini_api_key_here
SECRET_KEY=your_custom_internal_secret_key_here <== shared with backend

```



## How to Use

1. **Start the server** locally using Uvicorn:


```bash
uvicorn app:app --reload

```


The API will boot up and listen on `http://127.0.0.1:8000`.

2. **Open the API documentation**: `http://127.0.0.1:8000/docs` 

From Swagger UI, you can:

- View all available endpoints.
- Enter the required `x-internal-key` security header.
- Test the API directly using **Try it out**.
- View request and response schemas.

**Example Request Body:**

```json
{
  "category": "Computer Science",
  "sub_category": "Data Structures",
  "difficulty": "Hard",
  "language": "Arabic",
  "allowed_topics": ["Trees", "Graphs", "Hash Tables"],
  "num_questions": 5,
  "excluded_concepts": ["Binary Search Tree Complexity", "Graph BFS"]
}

```

**Example Successful Response:**

```json
{
  "quiz_title": "Advanced Algorithms Quiz",
  "questions": [
    {
      "question": "Which of the following is true about Dijkstra's algorithm?",
      "options": [
        "It handles negative weight edges effectively.",
        "It uses a priority queue to greedily select the closest unvisited node.",
        "It is fundamentally a divide-and-conquer algorithm.",
        "It computes the all-pairs shortest path."
      ],
      "correct_index": 1,
      "explanation": "Dijkstra's algorithm is a greedy algorithm that uses a priority queue to continuously find the shortest path from the source to all other nodes. It fails with negative edge weights.",
      "topic": "Graph Traversal",
      "concept_tag": "Hash Table Time Complexity"
    }
  ]
}

```