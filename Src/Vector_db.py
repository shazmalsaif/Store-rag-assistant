import faiss
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
# 1.Flatch raw data
from pathlib import Path

def load_md_files(folder_path: str) -> list[dict]:
    """Reads all .md files in a folder, returns list of {source, content}."""
    docs = []
    folder = Path(folder_path)

    for file_path in folder.glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        docs.append({
            "source": file_path.name,
            "content": content
        })

    return docs


# 2.Divide into chunk

def chunk_documents(docs: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Chunks every document, keeps track of which source + chunk index each piece came from."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    all_chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["content"])
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": f"{doc['source']}_{i}",
                "text": piece
            })
    return all_chunks


# 3.Generate Embedding & # 4.Store into vector store
def build_vector_store(chunks: list[dict], save_path: str = "Data/vector_data"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]

    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local(save_path)

    print(f"Saved {len(chunks)} chunks to FAISS index at '{save_path}'")
    return vector_store


if __name__ == "__main__":
    documents = load_md_files("Data/raw_data")
    print(f"Loaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    print(f"Total chunks: {len(chunks)}")

    build_vector_store(chunks)

