# back.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ FastAPI app object (name should match uvicorn command)
main = FastAPI()

# ✅ CORS setup to allow frontend access
main.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can replace "*" with your Streamlit frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Your Gemini API key (can be hardcoded for now)
API_KEY = "AIzaSyBzFr-G4_pZG_lxDrMDO1O3-n4WIkKHUUQ"  # ⚠ Don't commit this in GitHub public repo

# ✅ Route to handle question answering
@main.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    pdf_text = data.get("pdf_text")

    if not question or not pdf_text:
        return {"error": "Missing 'question' or 'pdf_text' in request"}

    # ✅ Step 1: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(pdf_text)

    # ✅ Step 2: Embed chunks and create FAISS vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # ✅ Step 3: Retrieve similar documents
    docs = vector_store.similarity_search(question)

    # ✅ Step 4: QA Prompt template
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

    # ✅ Step 5: Gemini QA chain
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=API_KEY
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    # ✅ Step 6: Run the chain
    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )

    # ✅ Step 7: Return final answer
    return {"answer": response["output_text"]}
