import os
from dotenv import load_dotenv
from typing import List

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Chroma and embedding
embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "Fynxt_database"

# Check if collection exists or create new
try:
    chroma_collection = chroma_client.get_collection(collection_name, embedding_function=embedding_function)
except:
    chroma_collection = chroma_client.create_collection(collection_name, embedding_function=embedding_function)

# Text wrapping utility (optional for CLI)
def word_wrap(text, width=100):
    import textwrap
    return "\n".join(textwrap.wrap(text, width))

# Build knowledgebase (run once or if PDFs are updated)
def build_pdf_knowledgebase(pdf_paths: List[str]):
    all_text = []

    for path in pdf_paths:
        reader = PdfReader(path)
        text_content = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
        all_text.extend(text_content)

    combined_text = '\n\n'.join(all_text)

    # Step 1: Character-level splitting
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=2000,
        chunk_overlap=200
    )
    character_chunks = character_splitter.split_text(combined_text)

    # Step 2: Token-level splitting
    token_splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=256, chunk_overlap=20)
    token_chunks = []
    for text in character_chunks:
        token_chunks += token_splitter.split_text(text)

    # Filter short/empty chunks
    token_chunks = [chunk for chunk in token_chunks if len(chunk.strip()) > 30]

    # Store in ChromaDB
    ids = [str(i) for i in range(len(token_chunks))]
    metadatas = [{"source": f"{os.path.basename(path)}_chunk_{i}"} for i in range(len(token_chunks))]
    chroma_collection.add(ids=ids, documents=token_chunks, metadatas=metadatas)

# Run once to build
build_pdf_knowledgebase(["G:/Fyxt_CS/Docs/Final_Doc.pdf"])

# Query function
def query_pdf_knowledgebase(query: str) -> str:
    """
    Answers a user's query based on pre-loaded documentation in ChromaDB.

    """
    model = genai.GenerativeModel("gemini-1.5-pro")
    results = chroma_collection.query(query_texts=[query], n_results=5)
    retrieved_documents = results['documents'][0]

    if not retrieved_documents:
        return "⚠️ No relevant information found."

    context = "\n\n".join([doc.strip() for doc in retrieved_documents])

    prompt = (
        "You are a helpful customer support assistant. Your users are asking questions about specific feature of the financial software."
        "Use only the provided content to answer accurately.\n\n"
        f"Question: {query}\n\n"
        f"Information:\n{context}"
    )

    response = model.generate_content(prompt)
    return response.text

