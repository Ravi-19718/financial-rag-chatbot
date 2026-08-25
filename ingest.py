import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()

# Folder where your PDFs live
REPORTS_DIR = "data/reports"

# Folder where ChromaDB will save the vector database
CHROMA_DIR = "chroma_db"


def ingest_documents():
    # --- STEP 1: Load all PDFs from the reports folder ---
    print("Loading PDF documents...")
    docs = []

    for filename in os.listdir(REPORTS_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(REPORTS_DIR, filename)
            print(f"  Reading: {filename}")
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())

    if not docs:
        print("No PDFs found in data/reports/. Please add PDF files and try again.")
        return

    print(f"Loaded {len(docs)} pages total.")

    # --- STEP 2: Split pages into smaller chunks ---
    # Why? Because LLMs have a limit on how much text they can process at once.
    # We break the document into chunks of ~1000 characters, with 200 characters
    # of overlap between chunks so we don't lose context at the edges.
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # --- STEP 3: Create embeddings ---
    # Embeddings convert text into numbers that capture meaning.
    # "all-MiniLM-L6-v2" is a free, fast model that runs locally on your machine.
    # First run will download the model (~90MB). After that it's cached.
    print("Creating embeddings (first run downloads the model — this takes a few minutes)...")
    embeddings = FastEmbedEmbeddings()

    # --- STEP 4: Store everything in ChromaDB ---
    # ChromaDB is a local vector database. It saves the chunks + their embeddings
    # to a folder called "chroma_db" so you don't have to re-process the PDFs
    # every time you start the chatbot.
    print("Storing embeddings in ChromaDB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"\nDone! Vector database saved to '{CHROMA_DIR}/'")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    ingest_documents()