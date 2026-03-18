import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ── 1. Load environment variables ──────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── 2. Load the embedding model ────────────────────────────────────────────
# Same model as ingestion — must match or retrieval breaks
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ── 3. Connect to the existing ChromaDB vector store ───────────────────────
# Loading existing DB, not recreating it
print("Connecting to ChromaDB...")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# k=4 means fetch the 4 most relevant chunks per question
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ── 4. Initialize Groq LLM ─────────────────────────────────────────────────
# llama-3.3-70b is Groq's most capable free model.
# temperature=0.2 keeps answers factual and consistent —
# lower temperature = less creative, more reliable. Right for legal content.
print("Initializing Groq LLM...")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.2
)

print("RAG pipeline ready.\n")


# ── 5. The core RAG function ───────────────────────────────────────────────
def ask(question: str, chat_history: list = []) -> dict:
    """
    Takes a question, retrieves relevant chunks from ChromaDB,
    and uses Groq/LLaMA to generate a grounded answer.

    Args:
        question: The user's question as a string
        chat_history: List of previous messages for context (optional)

    Returns:
        A dict with 'answer' and 'sources' keys
    """

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
    # This is prompt engineering — explicit instructions that ground
    # the model to your documents and define its behavior
    system_prompt = """You are a knowledgeable Nigerian tax law assistant.
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
""".format(context=context)

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