# 📄 RAG PDF Chatbot — Gemini + ChromaDB + HuggingFace

Ask questions about any PDF using **Retrieval-Augmented Generation (RAG)**.  
Powered by **Google Gemini**, **ChromaDB** vector store, and **HuggingFace** embeddings. Works locally or on Google Colab.

---

## 🧠 How It Works

```
Your PDF
   │
   ▼
Split into chunks (500 tokens)
   │
   ▼
Embed with all-MiniLM-L6-v2 (HuggingFace)
   │
   ▼
Store in ChromaDB (in-memory vector store)
   │
   ▼
User asks a question
   │
   ▼
Top 3 relevant chunks retrieved
   │
   ▼
Gemini generates a grounded answer
```

---

## ✨ Features

- 📄 Load and parse any PDF automatically
- 🔍 Semantic search with HuggingFace embeddings (runs on CPU, no GPU needed)
- 🤖 Gemini LLM generates accurate, context-grounded answers
- 💾 ChromaDB in-memory vector store (no setup required)
- 🔁 Fallback to retrieval-only mode if Gemini is unavailable
- 💬 Interactive Q&A loop after sample tests

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| LLM | Google Gemini (via `google-generativeai`) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, CPU) |
| Vector Store | ChromaDB |
| PDF Loader | LangChain `PyPDFLoader` |
| Framework | LangChain |

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Gemini API key

Get a free key at: https://aistudio.google.com/app/apikey

```bash
export GEMINI_API_KEY="your-key-here"
```

### 4. Add your PDF

```bash
mkdir data
# Copy your PDF into the data/ folder
cp your_document.pdf data/
```

### 5. Run

```bash
python rag_pdf_chatbot.py
```

---

## 📁 Project Structure

```
rag-pdf-chatbot/
│
├── rag_pdf_chatbot.py   # Main script
├── requirements.txt     # Python dependencies
├── data/                # Place your PDFs here (gitignored)
│   └── .gitkeep
└── README.md
```

---

## 🔐 API Key Safety

**Never hardcode your API key.** This project reads it from an environment variable:

```python
API_KEY = os.environ.get("GEMINI_API_KEY", "")
```

Add a `.env` file locally (never commit it):
```
GEMINI_API_KEY=your-key-here
```

And add to `.gitignore`:
```
.env
data/
```

---

## ☁️ Run on Google Colab

1. Upload `rag_pdf_chatbot.py` to Colab
2. Set your API key: `os.environ["GEMINI_API_KEY"] = "your-key"`
3. Upload PDFs to `/content/data/` using the file panel
4. Run the script

---

## 📄 License

MIT License — free to use and modify.
