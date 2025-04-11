# 🧠 LangGraph AI Support Agent with RAG and Ticketing System

This project implements a customer support agent using **LangGraph**, **ChromaDB**, and **Google Gemini 1.5 Pro**, enhanced with:
- 📄 **RAG (Retrieval-Augmented Generation)** to answer queries using internal documents.
- 🎟️ **Ticket Generation** when answers aren't confidently found.
- 🔁 A modular, memory-enabled structure using LangGraph tools.

---

## 🚀 Project Objectives

- Automate customer support using a smart agent.
- Use document-based retrieval for accurate responses.
- Create fallback tickets when knowledge-based answers are insufficient.

---

## 🛠️ How It Works

### 1. 🔐 Environment Setup
- Securely loads the `GOOGLE_API_KEY` using `getpass` or `.env`.
- Imports required modules: LangChain, LangGraph, ChromaDB, and more.

### 2. 📚 PDF Knowledge Base + ChromaDB
- Reads PDFs using `pypdf`.
- Splits text using `RecursiveCharacterTextSplitter`.
- Embeds text via `SentenceTransformerEmbeddingFunction`.
- Stores data in a persistent ChromaDB collection (`Fynxt_database`).

### 3. 🔍 RAG Tool: `query_pdf_knowledgebase`
- Takes a user query → finds matching content in ChromaDB.
- Sends context + query to **Gemini 1.5 Pro** for response generation.
- Returns:
  - Final answer
  - Boolean flag `relevant` (if useful data was found)

### 4. 🎟️ Ticket Tool: `create_ticket`
- If no relevant documents are found:
  - Generates a support ticket
  - Includes issue summary, suggested tags, and priority
- Designed for integration with external ticketing systems later

### 5. 🔄 LangGraph Workflow
- Uses `StateGraph` to define a clear flow:
  1. Human input
  2. → RAG tool
  3. → If not relevant → Ticket tool
- Uses memory to maintain message history between turns.

### 6. 🧪 Example Execution
- Demonstrates successful answers and fallback ticket generation depending on query relevance.

---

## 🧰 Tech Stack

| Tool/Library                  | Purpose                                 |
|------------------------------|-----------------------------------------|
| `LangGraph`                  | Manages agent flow                      |
| `ChromaDB`                   | Document retrieval via vector search    |
| `Google Gemini 1.5 Pro`      | Generates answers from context          |
| `LangChain`                  | RAG framework and message types         |
| `SentenceTransformers`       | Embedding generation for ChromaDB       |
| `pypdf`                      | PDF parsing                             |
| (Optional) `Gradio`          | Interface for interaction               |

---
