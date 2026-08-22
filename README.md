# Store RAG Assistant 🤖

A conversational **Retrieval-Augmented Generation (RAG) assistant for an e-commerce application**.

The assistant answers customer questions using information retrieved from the store's knowledge base instead of relying only on the language model's internal knowledge. It combines **FastAPI, LangGraph, FAISS, Hugging Face embeddings, and Llama 3.1 8B** to provide context-aware responses.

The backend is connected to a separate React/TypeScript e-commerce frontend.

**Frontend:** https://github.com/shazmalsaif/Hacakthon

**Backend:** https://github.com/shazmalsaif/Store-rag-assistant

---

# 🏗️ Architecture

```text
User
 │
 ▼
React E-commerce Frontend
 │
 │  POST /api/chat
 ▼
FastAPI
 │
 ▼
LangGraph
 │
 ├──────────────► Conversation Memory
 │
 ▼
RAG Retrieval
 │
 ▼
FAISS Vector Search
 │
 ▼
Relevant Store Knowledge
 │
 ▼
Llama 3.1 8B
 │
 ▼
Grounded Response
 │
 ▼
React Chatbot
```

The architecture combines a frontend chatbot with a backend RAG pipeline. User questions are sent to the FastAPI backend, where LangGraph manages the workflow and conversation state. Relevant information is retrieved from the FAISS vector store and provided to the LLM as context before generating the final response.

This allows the assistant to answer questions using the application's own product and store information.

---

# ✨ Features

* 🧠 Retrieval-Augmented Generation
* 🔍 Semantic search using FAISS
* 💬 Conversational question answering
* 🧵 Conversation state management using LangGraph Checkpointer
* ⚡ FastAPI REST API
* 🤗 Hugging Face models
* 🦙 Llama 3.1 8B Instruct
* 📚 Store and product knowledge retrieval
* 🌐 React frontend integration
* 🔐 Environment-based configuration
* ☁️ Backend deployment support

---

# 🧩 Technology Stack

| Component           | Technology                               |
| ------------------- | ---------------------------------------- |
| Backend API         | FastAPI                                  |
| AI Workflow         | LangGraph                                |
| Vector Search       | FAISS                                    |
| Embeddings          | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM                 | `meta-llama/Llama-3.1-8B-Instruct`       |
| Model Provider      | Hugging Face                             |
| Backend Language    | Python                                   |
| Frontend            | React + TypeScript + Vite                |
| Conversation Memory | LangGraph Checkpointer                   |
| Deployment          | Vercel                                   |

---

# 🔄 RAG Pipeline

The system has two major stages:

1. Knowledge Ingestion
2. Question Answering

## 1. Knowledge Ingestion

Store information is converted into searchable vector representations.

```text
Store Documents
      ↓
Document Processing
      ↓
Text Chunking
      ↓
Embedding Model
      ↓
Vector Embeddings
      ↓
FAISS
```

The embeddings are generated using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The resulting vectors are stored in the FAISS vector index and used during semantic retrieval.

---

## 2. Question Answering

When the user asks a question:

```text
User Question
      ↓
Query Processing
      ↓
Vector Retrieval
      ↓
FAISS
      ↓
Relevant Context
      ↓
LangGraph
      ↓
Llama 3.1 8B
      ↓
Final Answer
```

The retrieved information is provided to the language model as context so that the response is grounded in the store's knowledge base.

---

# 🧠 Conversation Memory

Conversation state is handled through **LangGraph's Checkpointer mechanism**.

The purpose of this layer is to allow the assistant to maintain context across multiple messages within a conversation.

For example:

```text
User:
What is the battery life of this speaker?

Assistant:
It has approximately 10 hours of battery life.

User:
Is it waterproof?

Assistant:
Yes, the speaker is rated IPX5...
```

The second question can be interpreted in the context of the previous conversation.

A `thread_id` is used to identify a conversation session:

```text
thread_id
    ↓
LangGraph Checkpointer
    ↓
Conversation State
```

---

# 🔌 API

## `POST /api/chat`

Sends a user message to the RAG assistant.

### Request

```json
{
  "message": "What is the battery life of this product?",
  "thread_id": "user-123"
}
```

### Response

```json
{
  "answer": "The product provides approximately 10 hours of battery life..."
}
```

---

## API Documentation

When running locally, FastAPI automatically provides interactive API documentation.

http://127.0.0.1:8000/docs

---

# 🚀 Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/shazmalsaif/Store-rag-assistant.git
cd Store-rag-assistant
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv myvenv
```

Activate it:

```powershell
.\myvenv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv myvenv
source myvenv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Hugging Face

Create a Hugging Face access token and add it to `.env`:

```env
HF_TOKEN=your_token_here
```

Do not commit your `.env` file to the repository.

---

## 5. Run the Backend

From the project root:

```bash
uvicorn api.index:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

http://127.0.0.1:8000/docs

---

# 🔗 Frontend Integration

The backend is designed to work with the separate e-commerce frontend.

```text
React + TypeScript Frontend
          │
          │ HTTPS / POST /api/chat
          ▼
FastAPI RAG Backend
          │
          ▼
RAG Assistant
```

The frontend sends requests to:

```text
/api/chat
```

with a message and conversation/thread identifier.

### Frontend Repository

https://github.com/shazmalsaif/Hacakthon

---

# 🌐 Deployment

The frontend and backend are maintained as separate repositories and can be developed and deployed independently.

The deployed architecture is:

```text
React + TypeScript Frontend
          │
          │ HTTPS / POST /api/chat
          ▼
FastAPI RAG Backend
          │
          ├── LangGraph
          ├── FAISS
          └── Hugging Face
```

This separation allows the frontend application and AI backend to be maintained independently while communicating through the REST API.

---

# 🎯 Project Goal

The goal of this project is to demonstrate how a modern e-commerce application can integrate a **Retrieval-Augmented Generation system** to provide users with a conversational interface for accessing product and store information.

Rather than building a chatbot that simply generates text, this project focuses on building a complete AI pipeline:

```text
Store Knowledge
      ↓
Document Processing
      ↓
Text Chunking
      ↓
Embeddings
      ↓
FAISS Vector Search
      ↓
Relevant Context
      ↓
LangGraph Workflow
      ↓
Llama 3.1 8B
      ↓
Context-Aware Response
      ↓
E-commerce Chatbot
```

The architecture demonstrates how retrieval, workflow orchestration, conversation state, and LLM generation can work together in a practical e-commerce application.

---

# 📄 Related Repository

## E-commerce Frontend

https://github.com/shazmalsaif/Hacakthon

This repository contains the e-commerce frontend and AI chatbot interface that communicates with this RAG backend.

---

# 👨‍💻 Author

**Shazamal Saif**

Computer Science & Engineering Student

GitHub: https://github.com/shazmalsaif
