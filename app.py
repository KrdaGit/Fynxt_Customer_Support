import os
import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = SentenceTransformerEmbeddingFunction()
collection = chroma_client.get_collection("Stockology_Article_1", embedding_function=embedding_function)

# Gemini model
model = genai.GenerativeModel("gemini-1.5-pro")

# Streamlit UI
st.set_page_config(page_title="Fynxt Customer Support", page_icon="🎧")

# Inject custom CSS & Header
st.markdown("""
    <style>
        body {
            background-color: #f4f6f8;
        }
        .title {
            font-size: 2.5rem;
            color: #4A90E2;
            text-align: center;
            margin-bottom: 1.5rem;
            font-weight: 700;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #555;
            margin-bottom: 2rem;
        }
        .chat-box {
            background-color: #fff;
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            max-width: 800px;
            margin: auto;
        }
        .response {
            background-color: #eaf4ff;
            padding: 1rem 1.2rem;
            margin-top: 1rem;
            border-left: 5px solid #4A90E2;
            border-radius: 8px;
            font-size: 1rem;
            color: #333;
            white-space: pre-wrap;
        }
        .footer {
            margin-top: 4rem;
            text-align: center;
            font-size: 0.85rem;
            color: #999;
        }
    </style>

    <div class="title">🧑‍💻 Fynxt Customer Support</div>
    <div class="subtitle">Ask anything related to our product or policy</div>
""", unsafe_allow_html=True)

#st.title("🧑‍💻 Fynxt Customer Support")
#st.markdown("Ask any question related to the our product or policy.")

with st.container():
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

    query = st.text_input("Enter your question:", key="user_query")

    if "run_query" not in st.session_state:
        st.session_state.run_query = False

    if query and not st.session_state.run_query:
        st.session_state.run_query = True

    if st.button("Ask"):
        st.session_state.run_query = True

    if st.session_state.run_query and query.strip():
        results = collection.query(query_texts=[query], n_results=5)
        retrieved_docs = results['documents'][0]

        if not retrieved_docs:
            st.warning("No relevant information found.")
        else:
            context = "\n\n".join([doc.strip() for doc in retrieved_docs])
            prompt = (
                "You are a helpful support assistant. Your users are asking questions about our product or policy documentation. "
                "Use only the provided content to answer accurately.\n\n"
                f"Question: {query}\n\n"
                f"Information:\n{context}"
            )

            response = model.generate_content(prompt)
            st.markdown(f"<div class='response'>{response.text}</div>", unsafe_allow_html=True)

        st.session_state.run_query = False

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <div class="footer">
        © 2025 Fynxt Technologies · All rights reserved
    </div>
""", unsafe_allow_html=True)
