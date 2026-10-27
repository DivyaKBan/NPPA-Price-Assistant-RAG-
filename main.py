from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer

app = FastAPI(title="RAG Assistant")

class Question(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(q: Question):
    return answer(q.question)