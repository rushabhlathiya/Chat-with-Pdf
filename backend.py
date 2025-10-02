import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage,AIMessage,HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langgraph.graph import StateGraph,START,END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from typing import TypedDict,Annotated

os.environ['HF_HOME'] = 'D:/huggingface_cache'

load_dotenv()

PDF_PATH = "islr.pdf"  # change to your file
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)



class ChatStateSchema(TypedDict):
    pdf_path: str 
    chunk_size: int 
    chunk_overlap: int 
    embed_model_name: str 
    force_rebuild: bool
    context:str
    message:Annotated[list[BaseMessage],add_messages]

graph = StateGraph(ChatStateSchema)
model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash-lite')


# ----------------- helpers (traced) -----------------
@traceable(name="load_pdf")
def load_pdf(path: str):
    return PyPDFLoader(path).load()  # list[Document]

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits, embed_model_name: str):
    emb = HuggingFaceEmbeddings(model=embed_model_name)
    return FAISS.from_documents(splits, emb)

# ----------------- cache key / fingerprint -----------------
def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}

def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str) -> str:
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()

# ----------------- explicitly traced load/build runs -----------------
@traceable(name="load_index", tags=["index"])
def load_index_run(index_dir: Path, embed_model_name: str):
    emb = HuggingFaceEmbeddings(model=embed_model_name)
    return FAISS.load_local(
        str(index_dir),
        emb,
        allow_dangerous_deserialization=True
    )

@traceable(name="build_index", tags=["index"])
def build_index_run(pdf_path: str, index_dir: Path, chunk_size: int, chunk_overlap: int, embed_model_name: str):
    docs = load_pdf(pdf_path)  # child
    print("docs",docs)
    print("chunk_size",chunk_size,"chunk_overlap",chunk_overlap)
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # child
    print(splits,embed_model_name)
    vs = build_vectorstore(splits, embed_model_name)  # child
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    (index_dir / "meta.json").write_text(json.dumps({
        "pdf_path": os.path.abspath(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
    }, indent=2))
    return vs

# ----------------- dispatcher (not traced) -----------------
def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "facebook/bart-base",
    force_rebuild: bool = False,
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key
    cache_hit = index_dir.exists() and not force_rebuild
    if cache_hit:
        return load_index_run(index_dir, embed_model_name)
    else:
        return build_index_run(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name)
    
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(pdf_path: str, chunk_size, chunk_overlap, embed_model_name="facebook/bart-base", force_rebuild=False):
    return load_or_build_index(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        force_rebuild=force_rebuild,
    )

@traceable(name="Context Finder")
def context_finder(state:ChatStateSchema):
    vectorstore = setup_pipeline(state['pdf_path'], state['chunk_size'], state['chunk_overlap'], state['embed_model_name'], state['force_rebuild'])
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    question = state['message'][-1].content
    docs =retriever.invoke(question)
    context = format_docs(docs)
    return {'context':context}

@traceable(name ='Chat Model')
def chat(state:ChatStateSchema):
    message_history = state['message']
    prompt = ChatPromptTemplate([
        ('placeholder', "{message_history}"),
        ('human',"Answer ONLY from the provided context. If not found, say you don't know. Based on the provided content and question\nquestion:{question}\n context:{context}")
        ])
    chain = prompt | model
    result= chain.invoke({'message_history':message_history[:-1],'question':message_history[-1].content,'context':state['context']})
    return {'message':[result]}



graph.add_node('context_finder',context_finder)
graph.add_node('chat',chat)

graph.add_edge(START,'context_finder')
graph.add_edge('context_finder','chat')
graph.add_edge('chat',END)

# conn =sqlite3.connect(database='chat.db',check_same_thread=False)

# checkpointer = SqliteSaver(conn=conn)

# workflow =graph.compile(checkpointer=checkpointer)
workflow =graph.compile()


# initial_state={
#     'pdf_path':'islr.pdf',
#     'chunk_size': 1000,
#     'chunk_overlap': 150,
#     'embed_model_name': "facebook/bart-base",
#     'force_rebuild': False,
#     'message':[HumanMessage('Who is the writer of this book')],
# }
# final_state =workflow.invoke(initial_state)

# print(final_state)

# final_state['message'][-1].content
# while True:
#     user_message =input("Chat Here:")

#     print(f'User:{user_message}')

#     if user_message.strip().lower() in ['exit','quit','bye']:
#         break
#     initial_state={
#         'pdf_path':'islr.pdf',
#         'chunk_size': 1000,
#         'chunk_overlap': 150,
#         'embed_model_name': "facebook/bart-base",
#         'force_rebuild': False,
#         'message':[HumanMessage(user_message)],
#         }
#     result = workflow.invoke(initial_state)

#     print(f'AI:{result['message'][-1].content}')
# I want you to create a streamlit UI for a chatbot when one asks quetion question as well as asnwer should be displayed like a chat there should be a opition to upload a pdf during chat there should be a side bar with a option of new chat and also showing the old chats each chat should have a feature of rename provide full code 
# there is a flow in this code there a a pdf upload interface which is on the top after the heading chatbot ui when i type first query it is on top of my first query chat but when i type some thing else like next question it changes it position to so it stays on the top of my latest question