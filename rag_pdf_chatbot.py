# ==========================================
# RAG PDF Chatbot with Gemini + ChromaDB
# ==========================================
# Upload your PDF to the /content/data folder (if using Colab)
# or place it in a local ./data/ folder.
# Then run this script.

# Install dependencies:
# pip install langchain langchain-community langchain-chroma
#             sentence-transformers pypdf langchain-huggingface
#             chromadb google-generativeai

import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# ==========================================
# STEP 1: CONFIGURE GEMINI API
# ==========================================

API_KEY = os.environ.get("GEMINI_API_KEY", "")  # Set via environment variable

if not API_KEY:
    raise ValueError(
        "❌ No API key found!\n"
        "Set your Gemini API key as an environment variable:\n"
        "  export GEMINI_API_KEY='your-key-here'\n"
        "Or get a free key at: https://aistudio.google.com/app/apikey"
    )

model = None
try:
    genai.configure(api_key=API_KEY)

    print("🔍 Checking available Gemini models...\n")
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
            print(f"✅ {m.name}")

    if not available_models:
        raise Exception("No models available. Check your API key!")

    model_name = available_models[0].replace('models/', '')
    print(f"\n🤖 Using model: {model_name}\n")
    model = genai.GenerativeModel(model_name)

except Exception as e:
    print(f"\n⚠️ Gemini API Error: {e}")
    print("👉 Falling back to retrieval-only mode (no LLM)\n")
    model = None


# ==========================================
# STEP 2: LOAD PDF DOCUMENTS
# ==========================================

DATA_FOLDER = "./data"
os.makedirs(DATA_FOLDER, exist_ok=True)


def load_docs(folder_path):
    docs = []
    if not os.path.exists(folder_path) or not os.listdir(folder_path):
        print("⚠️  No PDFs found! Place your PDF files in the ./data/ folder.")
        return []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            print(f"📄 Loading: {file}")
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())
    return docs


print("📦 Loading documents...")
docs = load_docs(DATA_FOLDER)

if not docs:
    print("\n❌ No documents loaded. Add PDFs to the ./data/ folder and re-run.")
    exit(1)

print(f"✅ Loaded {len(docs)} pages")


# ==========================================
# STEP 3: CHUNK + EMBED + VECTOR STORE
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = text_splitter.split_documents(docs)
print(f"✅ Created {len(chunks)} chunks")

print("🔍 Creating embeddings (this may take a moment)...")
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

texts = [c.page_content for c in chunks]

print("💾 Building vector store...")
db = Chroma(collection_name="rag_store", embedding_function=embedding_model)
db.add_texts(texts)
retriever = db.as_retriever(search_kwargs={"k": 3})

print("✅ RAG system ready!\n")


# ==========================================
# STEP 4: ANSWER FUNCTION
# ==========================================

def rag_answer(query: str) -> str:
    """Retrieve relevant chunks and generate an answer using Gemini."""
    print(f"🔍 Searching for: '{query}'")
    results = retriever.invoke(query)
    context = "\n\n".join([r.page_content for r in results])

    if model:
        prompt = f"""Based on the following context from a PDF document, answer the question concisely.

Context:
{context}

Question: {query}

Answer:"""
        print("🤖 Generating answer with Gemini...")
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini error: {e}")
            print("📋 Falling back to retrieved chunks:\n")
            return context
    else:
        print("📋 Retrieval-only mode — showing matched chunks:\n")
        return context


# ==========================================
# STEP 5: RUN SAMPLE QUERIES
# ==========================================

print("=" * 60)
print("TEST 1: Summary")
print("=" * 60)
answer1 = rag_answer("Give me a summary of the data in the PDF")
print(f"\n📝 Answer:\n{answer1}\n")

print("=" * 60)
print("TEST 2: Main Topics")
print("=" * 60)
answer2 = rag_answer("What are the main topics discussed?")
print(f"\n📝 Answer:\n{answer2}\n")


# ==========================================
# OPTIONAL: Interactive loop
# ==========================================

print("=" * 60)
print("💬 Ask your own questions! (type 'exit' to quit)")
print("=" * 60)
while True:
    user_query = input("\nYour question: ").strip()
    if user_query.lower() in ("exit", "quit", "q"):
        print("👋 Goodbye!")
        break
    if user_query:
        answer = rag_answer(user_query)
        print(f"\n📝 Answer:\n{answer}")
