from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Src.Connect_DB_with_LLM import (
    load_vector_store,
    get_llm,
    build_chat_graph
)

app = FastAPI(title="Store RAG Assistant API")


# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://hacakthon-liard.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load RAG components
vector_store = load_vector_store()
llm = get_llm()
chat_graph = build_chat_graph(vector_store, llm)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


@app.get("/api")
def root():
    return {"message": "Store RAG API is running"}


@app.post("/api/chat")
def chat(request: ChatRequest):

    result = chat_graph.invoke(
        {
            "messages": [
                ("user", request.message)
            ]
        },
        config={
            "configurable": {
                "thread_id": request.thread_id
            }
        }
    )

    return {
        "answer": result["messages"][-1].content
    }