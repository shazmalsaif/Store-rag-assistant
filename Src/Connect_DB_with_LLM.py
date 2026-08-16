from langgraph.graph import StateGraph,MessagesState,START
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage,HumanMessage
from dotenv import load_dotenv
import os
from Src.config import VECTOR_STORE_PATH, EMBEDDING_MODEL, LLM_MODEL
load_dotenv()

def load_vector_store(path: str = VECTOR_STORE_PATH):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True  
    )
    return vector_store

def retrieve_context( vector_store, query:str, k:int = 4)-> list[dict]:
    result = vector_store.similarity_search(query, k=k)
    return[{"text": r.page_content, "source": r.metadata.get("source","unknown")}
           for r in result]

def get_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        temperature=.3,
        task="conversational",
        )
    return ChatHuggingFace(llm=endpoint)

def build_chat_graph(vector_store,llm):
    def call_model( state:MessagesState):
        last_user_msg = state["messages"][-1].content

        chunks = retrieve_context(vector_store,last_user_msg)
        context ="\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
        system_msg =SystemMessage(content=f"""You are a helpful customer support assistant for an online gadget store.
Use the conversation so far to understand follow-up questions. Answer using ONLY the context below —
if the answer isn't there, say you don't have that information.

Context:
{context}
""")
        response = llm.invoke([system_msg]+state["messages"])
        return{"messages":[response]}

    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model",call_model)
    workflow.add_edge(START,"model")
    checkpoint = MemorySaver()
    return workflow.compile(checkpointer=checkpoint)

if __name__ == "__main__":
    vector_store = load_vector_store()
    llm = get_llm()
    app = build_chat_graph(vector_store,llm)
    config = {"configurable":{"thread_id":"session-1"}}
    print("chatbot is ready.Type 'quit' to exit.\n")
    while True:
        query = input("you: ")
        if query.lower() == "quit":
            break

        result = app.invoke({"messages":[HumanMessage(content=query)]},config=config)
        print(f"\nBot: {result['messages'][-1].content}\n")
        