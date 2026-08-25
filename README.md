# 📊 Financial Reports RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers natural language questions about 5 Fortune 50 company annual reports (10-K filings). Built with LangChain, ChromaDB, and Groq — deployed live on Streamlit Cloud.

🔗 **[Live Demo](https://financial-rag-chatbot-cjwfwegkze9fqdn4xspjeh.streamlit.app/)**

---

## 🧠 How It Works

```
PDF Reports → Text Chunking → Embeddings → ChromaDB Vector Store
                                                      ↓
User Question → Semantic Search → Relevant Chunks → Groq LLM → Answer
```

1. **Ingestion** — 10-K PDFs are parsed, split into overlapping chunks, and embedded using FastEmbed (ONNX-based, no GPU required)
2. **Storage** — Embeddings are stored in ChromaDB, a local persistent vector database
3. **Retrieval** — On each query, the top 4 most semantically similar chunks are retrieved
4. **Generation** — Retrieved context + user question are passed to Groq's LLM to generate a grounded answer

---

## 🏢 Covered Companies (Fortune 50)

| Company | Filing | Source |
|---------|--------|--------|
| 🛒 Walmart | 10-K Annual Report | SEC EDGAR |
| ☁️ Amazon | 10-K Annual Report | SEC EDGAR |
| 🍎 Apple | 10-K Annual Report | SEC EDGAR |
| 🪟 Microsoft | 10-K Annual Report | SEC EDGAR |
| 🏦 MetLife | 10-K Annual Report | SEC EDGAR |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit |
| **LLM** | Groq API (fast inference) |
| **Orchestration** | LangChain LCEL |
| **Embeddings** | FastEmbed (ONNX) |
| **Vector DB** | ChromaDB |
| **PDF Parsing** | PyPDF |
| **Deployment** | Streamlit Cloud |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.9+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup

```bash
# Clone the repo
git clone https://github.com/Ravi-19718/financial-rag-chatbot.git
cd financial-rag-chatbot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Add your API key

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### Add PDF reports

Download 10-K filings from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar) and place them in:
```
data/reports/
```

### Ingest documents and run

```bash
# Build the vector database
python ingest.py

# Launch the app
streamlit run app.py
```

---

## 📁 Project Structure

```
financial-rag-chatbot/
├── app.py              # Streamlit app and RAG chain
├── ingest.py           # PDF ingestion and embedding pipeline
├── requirements.txt    # Python dependencies
├── chroma_db/          # Persisted vector database
├── data/
│   └── reports/        # PDF source documents (not tracked in git)
└── .env                # API keys (not tracked in git)
```

---

## 💡 Example Questions

- *"What was Walmart's total revenue in fiscal year 2024?"*
- *"How does Apple describe its competitive landscape?"*
- *"What are Microsoft's key risk factors?"*
- *"Compare Amazon and MetLife's net income."*

---

## 🔑 Key Design Decisions

- **FastEmbed over HuggingFace** — avoids PyTorch/NumPy version conflicts, runs on CPU with no setup friction
- **LangChain LCEL** — modern chain composition replaces deprecated `RetrievalQA`, cleaner and more maintainable
- **ChromaDB committed to repo** — enables zero-setup deployment on Streamlit Cloud without a rebuild step
- **`st.cache_resource`** — vector store and LLM load once per session, not on every message

---

## 📄 License

MIT
