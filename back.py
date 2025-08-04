from fastapi import FastAPI
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

# ✅ Global vector store (set on startup)
vector_store = None

# ✅ Load the PDF on startup and create vector index
@main.on_event("startup")
def load_policy_pdf():
    global vector_store

    # Load and extract text from PDF
    pdf_path = "Arogya_Sanjeevani.pdf"  # You should copy the uploaded PDF here
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"

    # Chunk and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(full_text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )

    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

# ✅ Input model
class AskRequest(BaseModel):
    question: str

# ✅ Health check
@main.get("/")
def root():
    return {"message": "API is running"}

# ✅ Ask endpoint
@main.post("/ask")
async def ask_question(body: AskRequest):
    global vector_store
    question = body.question

    # Retrieve relevant chunks
    docs = vector_store.similarity_search(question)

    # Prompt template
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

    # Gemini Model
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=API_KEY
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    # Run chain
    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )

    return {"answer": response["output_text"]}
