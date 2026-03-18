from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .rag import ask

#Initialize the FastAPI app
app = FastAPI(
    title="Nigerian Tax Law Chatbot API",
    description="A RAG-based API for answering Nigerian tax law questions",
    version="1.0.0"
)

#CORS middleware
# CORS (Cross-Origin Resource Sharing) controls which frontends are allowed to talk to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Define request and response models
# Pydantic models define the shape of data coming in and going out.
class ChatRequest(BaseModel):
    question: str
    chat_history: list = []

class ChatResponse(BaseModel):
    answer: str
    sources: list

#Define the endpoints

@app.get("/health")
def health_check():
    """
    A simple endpoint to confirm the server is running.
    Useful for debugging and monitoring in production.
    """
    return {"status": "ok", "message": "Nigerian Tax Chatbot API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint. Receives a question, runs the RAG pipeline,
    and returns an answer grounded in Nigerian tax law documents.
    """
    result = ask(
        question=request.question,
        chat_history=request.chat_history
    )
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"]
    )