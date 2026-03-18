import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

#Load all .txt files from data/processed
print("Loading documents...")
loader = DirectoryLoader(
    "data/processed",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()
print(f"  ✓ Loaded {len(documents)} documents")

#Split documents into chunks 
print("Chunking documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"  ✓ Created {len(chunks)} chunks")

#Create embeddings and store in ChromaDB
print("Loading embedding model...")
print("  (First run will download the model — ~90MB, only happens once)")
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
print("  ✓ Embedding model loaded")

# Chroma.from_documents embeds every chunk and stores both the text
print("Embedding chunks and storing in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print(f"  ✓ Embedded and stored {len(chunks)} chunks in ChromaDB")
print("\nIngestion complete. Your vector store is ready.")