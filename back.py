from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pdfplumber
import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ FastAPI app object
main = FastAPI()

# ✅ CORS
main.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Gemini API key (⚠ Do not expose in prod)
API_KEY = "AIzaSyBzFr-G4_pZG_lxDrMDO1O3-n4WIkKHUUQ"

# ✅ Global vector store
vector_store = None

# ✅ Load the PDF on startup and create vector index
@main.on_event("startup")
def load_policy_pdf():
    global vector_store

    pdf_path = "Arogya_Sanjeevani.pdf"
    full_text = ""

    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(full_text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )

    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    print(f"✅ PDF loaded with {len(chunks)} chunks.")

# ✅ Health check
@main.get("/")
def root():
    return {"message": "API is running"}

# ✅ Combined GET + POST route for /ask
@main.api_route("/ask", methods=["GET", "POST"])
async def ask_question(request: Request):
    global vector_store

    if request.method == "GET":
        return {
            "message": "👋 Welcome to /ask endpoint. Please send a POST request with a JSON body like: { 'question': '...' }"
        }

    try:
        body = await request.json()
    except Exception:
        return {"answer": "❌ Invalid JSON body."}

    question = body.get("question", "").strip()

    if not question:
        return {"answer": "❌ No question provided."}

    if vector_store is None:
        return {"answer": "❌ PDF not loaded yet. Please try again later."}

    docs = vector_store.similarity_search(question)

    prompt_template = """
    Answer the question as accurately as possible using the context below.
    If the answer is not available, respond with: "Answer not available in the context."

    Context:
    {context}

    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=API_KEY
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )

    return {"answer": response["output_text"]}
