import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load API keys from .env
load_dotenv()

CHROMA_DIR = "chroma_db"

# --- Page setup ---
st.set_page_config(
    page_title="Financial Reports Chatbot",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Financial Reports RAG Chatbot")
st.caption("Ask questions about public financial reports — powered by Groq + LangChain")

# --- Load the RAG chain (cached so it only loads once) ---
# @st.cache_resource means Streamlit loads this once and reuses it,
# instead of reloading the model on every message.
@st.cache_resource
def load_chain():
    # Load the same embedding model used during ingestion
    embeddings = FastEmbedEmbeddings()

    # Connect to the ChromaDB we built with ingest.py
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    # Connect to Groq's LLM — llama-3.1-8b-instant is fast and free
    llm = ChatGroq(
        model="groq/compound-mini",
        temperature=0,  # 0 = factual, consistent answers (no creativity)
        api_key=os.getenv("GROQ_API_KEY")
    )

    # This is the instruction we give the LLM.
    # {context} = the relevant chunks retrieved from ChromaDB
    # {question} = what the user typed
    prompt_template = """You are a financial analyst assistant.
Use the following excerpts from financial reports to answer the question accurately and concisely.
If the answer is not found in the context, say "I couldn't find that information in the documents."

Context from financial reports:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # RetrievalQA ties everything together:
    # 1. Takes the user's question
    # 2. Retrieves the 4 most relevant chunks from ChromaDB (k=4)
    # 3. Feeds them + the question into the LLM using our prompt
    # 4. Returns the answer
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain


# --- Safety check: make sure ingest.py has been run first ---
if not os.path.exists(CHROMA_DIR):
    st.error("⚠️ No vector database found. Please run `python ingest.py` first.")
    st.stop()

# Load the chain
chain = load_chain()

# --- Chat interface ---
# st.session_state keeps the conversation history while the app is open
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wait for user input
if question := st.chat_input("Ask a question about the financial reports..."):

    # Show the user's question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and show the answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = chain.invoke({"query": question})
            answer = result["result"]
            sources = result["source_documents"]

        st.markdown(answer)

        # Show which pages the answer came from
        if sources:
            with st.expander("📄 View sources"):
                for i, doc in enumerate(sources[:3]):
                    filename = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page", 0)
                    st.markdown(f"**Source {i+1}:** {filename} — Page {page + 1}")
                    st.markdown(f"*...{doc.page_content[:250]}...*")
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})