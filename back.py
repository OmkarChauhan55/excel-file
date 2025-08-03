from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pdfplumber
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ FastAPI app object
main = FastAPI()

# ✅ CORS settings
main.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Gemini API key
API_KEY = "AIzaSyBzFr-G4_pZG_lxDrMDO1O3-n4WIkKHUUQ"

# ✅ Global vector store
vector_store = None

# ✅ Load the PDF and prepare embeddings on startup
@app.on_event("startup")
def load_pdf_and_embed():
    global vector_store

    pdf_path = "Arogya_Sanjeevani.pdf"
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(full_text)

    # Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )

    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

# ✅ Input model
class RunRequest(BaseModel):
    questions: list[str]

# ✅ Prompt
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

# ✅ QA Chain
def get_qa_chain():
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=API_KEY
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# ✅ Health check
@app.get("/")
def health():
    return {"status": "running"}

# ✅ Final webhook for HackRx
@app.post("/api/v1/hackrx/run")
async def hackrx_run(body: RunRequest):
    global vector_store
    chain = get_qa_chain()

    answers = []
    for question in body.questions:
        docs = vector_store.similarity_search(question)
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        answers.append(response["output_text"])

    return {"answers": answers}
