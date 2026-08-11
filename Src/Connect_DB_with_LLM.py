from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = "Data/vector_data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # swap for whichever HF model you're using


def load_vector_store(path: str = VECTOR_STORE_PATH):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True  # safe here — it's your own local index
    )
    return vector_store


def retrieve_context(vector_store, query: str, k: int = 4) -> list[dict]:
    """Returns top-k chunks with their source metadata."""
    results = vector_store.similarity_search(query, k=k)
    return [
        {"text": r.page_content, "source": r.metadata.get("source", "unknown")}
        for r in results
    ]


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"""You are a helpful customer support assistant for an online gadget store.
Answer the user's question using ONLY the context below. If the answer isn't in the context, say you don't have that information — do not make anything up.

Context:
{context}

Question: {query}

Answer:"""


def get_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        max_new_tokens=512,
        temperature=0.3,
    )
    return ChatHuggingFace(llm=endpoint)


def ask(query: str, vector_store, llm) -> dict:
    chunks = retrieve_context(vector_store, query)
    prompt = build_prompt(query, chunks)

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": list({c["source"] for c in chunks})  # unique source files used
    }


if __name__ == "__main__":
    vector_store = load_vector_store()
    llm = get_llm()

    print("Chatbot ready. Type 'quit' to exit.\n")
    while True:
        query = input("You: ")
        if query.lower() == "quit":
            break

        result = ask(query, vector_store, llm)
        print(f"\nBot: {result['answer']}")
        print(f"(Sources: {', '.join(result['sources'])})\n")
