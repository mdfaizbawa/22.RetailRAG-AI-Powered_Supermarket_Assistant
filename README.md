=========================================
🛒 RetailRAG – AI-Powered Supermarket Assistant
=========================================

RetailRAG is a full-stack Retrieval-Augmented Generation (RAG) based conversational AI system designed to answer supermarket product queries using semantic search and large language models.

This project demonstrates practical implementation of:

Vector Search

Context-Aware LLM Responses

FastAPI Backend

Streamlit Frontend

Session-Based Chat Memory

=============================
🚀 Project Overview
=========================================

RetailRAG allows users to ask natural language questions such as:

Do you have Tata Salt in stock?

What is the price of Amul milk?

Is basmati rice available?

Show available dairy products.

The system retrieves relevant product information from a vector database and generates accurate responses grounded in retrieved context.

=============================
🧠 Tech Stack
=========================================
🔹 Backend

Python

FastAPI

LangChain

ChromaDB (Vector Database)

Sentence Transformers (Embeddings)

HuggingFace Inference API (Mistral-7B)

🔹 Frontend

Streamlit

=============================
🔎 How It Works (RAG Pipeline)
=========================================

Load supermarket dataset (CSV).

Convert each row into LangChain Document objects.

Split documents into chunks using RecursiveCharacterTextSplitter.

Generate embeddings.

Store embeddings in Chroma Vector Database.

User query triggers semantic similarity search.

Top relevant chunks are retrieved.

Retrieved context + user question sent to LLM.

LLM generates grounded response.

The model is restricted to answer only from retrieved context, reducing hallucinations.

=============================
🔮 Future Enhancements
=========================================

Persistent Vector Database

Hybrid Search (semantic + filter)

Multilingual Support (English + Tamil)

JWT Authentication

Docker Deployment

Cloud Deployment (Render / Railway / Streamlit Cloud)

=============================
👨‍💻 Author
=========================================

MOHAMED FAIZ

GenAI & Data Science Developer

Thank you.
