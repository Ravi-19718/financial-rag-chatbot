import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = "chroma_db"

# Load API key — fail clearly if missing
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
    st.error("❌ GROQ_API_KEY is missing or invalid. Go to App Settings → Secrets and add it.")
    st.stop()

st.set_page_config(page_title="Financial Reports Chatbot", page_icon="📊", layout="centered")
st.title("📊 Financial Reports RAG Chatbot")
st.caption("Ask questions about Fortune 50 annual reports — powered by Groq + LangChain")

# Company coverage panel
with st.expander("📋 Covered Companies — click to expand", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🛒 **Walmart**")
        st.markdown("☁️ **Amazon**")
    with col2:
        st.markdown("🍎 **Apple**")
        st.markdown("🪟 **Microsoft**")
    with col3:
        st.markdown("🏦 **MetLife**")
    st.markdown("---")
    st.markdown("*Source: Annual 10-K filings from SEC EDGAR*")

st.divider()


@st.cache_resource
def load_chain():
    embeddings = FastEmbedEmbeddings()
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    llm = ChatGroq(model="groq/compound-mini", temperature=0, api_key=GROQ_API_KEY)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""You are a financial analyst assistant.
Use the following excerpts from financial reports to answer the question accurately and concisely.
If the answer is not found in the context, say "I couldn't find that information in the documents."

Context:
{context}

Question: {question}

Answer:""")

    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


if not os.path.exists(CHROMA_DIR):
    st.error("⚠️ No vector database found. Please run `python ingest.py` first.")
    st.stop()

chain, retriever = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about the financial reports..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            sources = retriever.invoke(question)
            answer = chain.invoke(question)
        st.markdown(answer)
        if sources:
            with st.expander("📄 View sources"):
                for i, doc in enumerate(sources[:3]):
                    filename = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page", 0)
                    st.markdown(f"**Source {i+1}:** {filename} — Page {page + 1}")
                    st.markdown(f"*...{doc.page_content[:250]}...*")
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})