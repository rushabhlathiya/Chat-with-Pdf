import streamlit as st
from backend import workflow
from langchain_core.messages import HumanMessage
import os
import uuid
import json

# Directories for storing data
CHAT_DIR = "chat_sessions"
PDF_DIR = "uploaded_pdfs"
os.makedirs(CHAT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

st.set_page_config(page_title="Chatbot UI", layout="wide")

# ----------- Utility Functions ------------
def load_chats():
    chats = []
    for file in os.listdir(CHAT_DIR):
        if file.endswith(".json"):
            with open(os.path.join(CHAT_DIR, file), "r") as f:
                data = json.load(f)
                chats.append({"id": file.replace(".json", ""), "name": data.get("name", "Unnamed Chat")})
    return chats

def save_chat(chat_id, chat_data):
    with open(os.path.join(CHAT_DIR, f"{chat_id}.json"), "w") as f:
        json.dump(chat_data, f)

def load_chat(chat_id):
    try:
        with open(os.path.join(CHAT_DIR, f"{chat_id}.json"), "r") as f:
            return json.load(f)
    except:
        return {"name": "New Chat", "messages": []}

def reset_chat():
    new_id = str(uuid.uuid4())
    st.session_state.chat_id = new_id
    st.session_state.chat_data = {
        "name": "New Chat",
        "messages": [],
        "pdf": None  # This will store PDF file name or path
    }
    save_chat(new_id, st.session_state.chat_data)

def rename_chat(chat_id, new_name):
    data = load_chat(chat_id)
    data["name"] = new_name
    save_chat(chat_id, data)

def delete_chat(chat_id):
    path = os.path.join(CHAT_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        os.remove(path)

def save_uploaded_pdf(uploaded_file):
    file_path = os.path.join(PDF_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# ----------- Initial Setup (Fix Reload Issue) ------------
if "initialized" not in st.session_state:
    saved_chats = load_chats()
    if saved_chats:
        st.session_state.chat_id = saved_chats[0]["id"]
        st.session_state.chat_data = load_chat(st.session_state.chat_id)
    else:
        reset_chat()
    st.session_state.initialized = True

# Rename tracking
if "rename_chat_id" not in st.session_state:
    st.session_state.rename_chat_id = None

# ----------- Sidebar UI ------------
st.sidebar.title("📁 Chat Sessions")

if st.sidebar.button("➕ New Chat"):
    reset_chat()
    st.session_state.rename_chat_id = None
    st.rerun()

chats = load_chats()
for chat in chats:
    if st.session_state.rename_chat_id == chat["id"]:
        with st.sidebar:
            new_name = st.text_input("Rename chat", chat["name"], key=f"input_{chat['id']}")
            save_col, cancel_col = st.columns([1, 1])
            if save_col.button("Save", key=f"save_rename_{chat['id']}"):
                rename_chat(chat["id"], new_name)
                st.session_state.rename_chat_id = None
                st.rerun()
            if cancel_col.button("Cancel", key=f"cancel_rename_{chat['id']}"):
                st.session_state.rename_chat_id = None
                st.rerun()
    else:
        col1, col2, col3 = st.sidebar.columns([4, 1, 1])
        if col1.button(chat["name"], key=f"load_{chat['id']}"):
            st.session_state.chat_id = chat["id"]
            st.session_state.chat_data = load_chat(chat["id"])
            st.session_state.rename_chat_id = None
            st.rerun()
        if col2.button("✏️", key=f"rename_{chat['id']}"):
            st.session_state.rename_chat_id = chat["id"]
            st.rerun()
        if col3.button("🗑", key=f"delete_{chat['id']}"):
            delete_chat(chat["id"])
            if chat["id"] == st.session_state.chat_id:
                saved = load_chats()
                if saved:
                    st.session_state.chat_id = saved[0]["id"]
                    st.session_state.chat_data = load_chat(st.session_state.chat_id)
                else:
                    reset_chat()
            st.session_state.rename_chat_id = None
            st.rerun()

# ----------- Main Chat UI ------------

st.title("💬 Chatbot UI")

# Load current chat data
chat_data = st.session_state.chat_data

# 🟡 Check if PDF is uploaded in this session
if chat_data.get("pdf") is None:
    uploaded_file = st.file_uploader("📄 Upload a PDF to begin chatting", type="pdf")
    if uploaded_file:
        saved_path = save_uploaded_pdf(uploaded_file)
        chat_data["pdf"] = saved_path  # Save path in chat session
        st.session_state.chat_data = chat_data
        save_chat(st.session_state.chat_id, chat_data)
        st.success(f"✅ PDF uploaded successfully: {uploaded_file.name}")
        st.rerun()
    else:
        st.warning("⚠ Please upload a PDF before starting the chat.")
else:
    st.info(f"📎 Chatting based on uploaded PDF: {os.path.basename(chat_data['pdf'])}")

     
    for msg in chat_data["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])


    # Chat input (enabled only after PDF upload)
    if prompt := st.chat_input("Ask a question"):
        st.chat_message("user").markdown(prompt)
        chat_data["messages"].append({"role": "user", "content": prompt})

        pdf_path = f"uploaded_pdfs\\{os.path.basename(chat_data['pdf'])}"
        initial_state={
            'pdf_path':pdf_path,
            'chunk_size': 1000,
            'chunk_overlap': 150,
            'embed_model_name': "facebook/bart-base",
            'force_rebuild': False,
            'message':[HumanMessage(prompt)],
        }
        # configurable={'configurable':{'thread_id':st.session_state['chat_id']}}

        ai_message =workflow.invoke(initial_state)



        st.chat_message("assistant").markdown(ai_message['message'][-1].content)
        chat_data["messages"].append({"role": "assistant", "content": ai_message['message'][-1].content})
        st.session_state.chat_data = chat_data
        save_chat(st.session_state.chat_id, chat_data)