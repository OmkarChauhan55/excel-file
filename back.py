from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ FastAPI app object
main = FastAPI()

# ✅ CORS middleware setup
main.add_middleware(
    CORSMiddleware,
    allow_origins=[""],  # Replace "" with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Gemini API key (⚠ Never expose this in public repos)
API_KEY = "AIzaSyBzFr-G4_pZG_lxDrMDO1O3-n4WIkKHUUQ"

# ✅ Request schema for /ask
class AskRequest(BaseModel):
    question: str
    pdf_text: str

# ✅ Health check (optional)
@main.get("/")
def root():
    return {"message": "API is running"}

# ✅ Question-answering endpoint
@main.post("/ask")
async def ask_question(body: AskRequest):
    question = body.question
    pdf_text = body.pdf_text

    # Step 1: Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(pdf_text)

    # Step 2: Embed and create FAISS store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # Step 3: Retrieve similar documents
    docs = vector_store.similarity_search(question)

    # Step 4: Prompt
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say "answer is not available in the context".

    Context:
    {context}

    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Step 5: Load Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=API_KEY
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    # Step 6: Run the chain
    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )

    # Step 7: Return answer
    return {"answer": response["output_text"]}
