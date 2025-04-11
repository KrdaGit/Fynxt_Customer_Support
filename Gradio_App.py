import os
import gradio as gr
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

# Query handling function
def chat_with_gemini(query):
    if not query.strip():
        return "Please enter a question."

    results = collection.query(query_texts=[query], n_results=5)
    retrieved_docs = results['documents'][0]

    if not retrieved_docs:
        return "⚠️ No relevant information found."

    context = "\n\n".join([doc.strip() for doc in retrieved_docs])
    prompt = (
        "You are a helpful support assistant. Your users are asking questions about our product or policy documentation. "
        "Use only the provided content to answer accurately.\n\n"
        f"Question: {query}\n\n"
        f"Information:\n{context}"
    )

    response = model.generate_content(prompt)
    return response.text

# Custom CSS
css = """
#chatbox {background-color: #fff; padding: 20px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.05);}
#title {font-size: 2rem; color: #4A90E2; font-weight: 700; text-align: center;}
#subtitle {text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 1rem;}
footer {margin-top: 3rem; text-align: center; font-size: 0.85rem; color: #999;}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("<div id='title'>🧑‍💻 Fynxt Customer Support</div>")
    gr.Markdown("<div id='subtitle'>Ask anything related to our product or policy</div>")

    with gr.Box(elem_id="chatbox"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Enter your question and press Enter", show_label=False)

        def respond(message, history):
            answer = chat_with_gemini(message)
            history.append((message, answer))
            return history, ""

        msg.submit(respond, [msg, chatbot], [chatbot, msg])

    gr.Markdown("<footer>© 2025 Fynxt Technologies · All rights reserved</footer>")

# Launch the app
demo.launch(share=True)
