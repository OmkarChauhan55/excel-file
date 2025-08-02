# app_cli.py
from rag_pipeline import load_pdf, split_text, get_vector_store, get_conversational_chain
import os

def main():
    print("\n=== PDF RAG CLI Chatbot ===")
    pdf_path = input("Enter path to your PDF file: ").strip()

    if not os.path.exists(pdf_path):
        print("[Error] File not found.")
        return

    # Load API Key (assumes Gemini)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Enter your Gemini API Key: ").strip()

    model_name = "Gemini"  # You can change this to OpenAI if needed

    print("\n[1/4] Reading PDF...")
    raw_text = load_pdf(pdf_path)

    print("[2/4] Splitting text into chunks...")
    chunks = split_text(raw_text)

    print("[3/4] Creating vector store...")
    vector_store = get_vector_store(chunks, model_name, api_key)

    print("[4/4] Initializing chatbot...")
    qa_chain = get_conversational_chain(vector_store, model_name, api_key)

    history = []
    print("\nChatbot ready. Ask anything about the PDF (type 'exit' to quit):")
    while True:
        question = input("\nYou: ")
        if question.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            response = qa_chain({"question": question, "chat_history": history})
            print("Bot:", response["answer"])
            history.append((question, response["answer"]))
        except Exception as e:
            print("[Error]", e)

if __name__ == "__main__":
    print("✅ Inside __main__")  # ← Add this for debug
    main()

print("✅ app_cli.py started")  # ← Add this at very top of your file
