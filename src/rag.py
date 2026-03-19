import os
import sys
from dotenv import load_dotenv

# Wrap everything in a try block to catch and print any import/init errors
try:
    print("🚀 1. Starting rag.py initialization...")

    # ── 1. Load environment variables ──────────────────────────────────────────
    print("📁 2. Loading .env file...")
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("❌ 3. ERROR: GROQ_API_KEY not found in environment variables!")
        print("    Please set it in Render dashboard → Environment tab")
        sys.exit(1)
    else:
        print("✅ 3. GROQ_API_KEY loaded successfully")

    # ── 2. Load the embedding model ────────────────────────────────────────────
    print("🧠 4. Importing HuggingFaceEmbeddings...")
    from langchain_huggingface import HuggingFaceEmbeddings

    print("🧠 5. Loading embedding model (all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ 6. Embedding model loaded")

    # ── 3. Connect to ChromaDB ─────────────────────────────────────────────────
    print("🔗 7. Importing Chroma...")
    from langchain_community.vectorstores import Chroma

    print("🔗 8. Connecting to ChromaDB at 'chroma_db'...")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("✅ 9. ChromaDB connected, retriever created")

    # ── 4. Initialize Groq LLM ─────────────────────────────────────────────────
    print("🤖 10. Importing ChatGroq...")
    from langchain_groq import ChatGroq

    print("🤖 11. Initializing Groq LLM (llama-3.3-70b)...")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2
    )
    print("✅ 12. Groq LLM initialized")

    print("🎉 13. RAG pipeline ready.\n")

except Exception as e:
    print(f"💥 FATAL ERROR in rag.py: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ── 5. The core RAG function (imports inside to avoid load-time crashes) ──
def ask(question: str, chat_history: list = []) -> dict:
    """
    Takes a question, retrieves relevant chunks from ChromaDB,
    and uses Groq/LLaMA to generate a grounded answer.
    """
    # Import these inside the function – they are only needed when the function is called
    from langchain_core.messages import HumanMessage, SystemMessage

    # Step A: Retrieve relevant chunks
    relevant_chunks = retriever.invoke(question)

    # Step B: Format chunks into a context string
    context = ""
    sources = []
    for i, chunk in enumerate(relevant_chunks):
        context += f"\n\n[Source {i+1}]\n{chunk.page_content}"
        source = chunk.metadata.get("source", "Unknown")
        if source not in sources:
            sources.append(source)

    # Step C: Build the system prompt
    system_prompt = f"""You are a knowledgeable Nigerian tax law assistant.
Your job is to answer questions about Nigerian tax laws accurately and clearly.

RULES:
1. Answer ONLY based on the context provided below. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question,
   say: "I don't have enough information in my documents to answer that question."
3. Be clear and concise. Use plain English where possible.
4. When referencing specific sections or Acts, mention them by name.
5. Do not give legal advice — remind users to consult a tax professional
   for specific situations.

CONTEXT FROM NIGERIAN TAX LAW DOCUMENTS:
{context}
"""

    # Step D: Build the message list with history
    messages = [SystemMessage(content=system_prompt)]

    for turn in chat_history:
        messages.append(HumanMessage(content=turn["question"]))
        messages.append(SystemMessage(content=turn["answer"]))

    messages.append(HumanMessage(content=question))

    # Step E: Call the LLM
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": sources
    }
