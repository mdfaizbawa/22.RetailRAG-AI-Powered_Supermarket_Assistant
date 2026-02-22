# backend/main.py

import pandas as pd
from google import genai

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# =========================================
# 🔐 API KEY
# =========================================

google_api_key = "******"
client = genai.Client(api_key=google_api_key)

print("API key loaded")

# =========================================
# 📂 LOAD DATA
# =========================================

df = pd.read_csv("supermarket_rag_dataset.csv")

documents = []

for _, row in df.iterrows():
    content = f"""
Product: {row['title']}
Category: {row['category']}
Brand: {row['brand']}
Price: {row['price_inr']} INR
Stock: {row['stock']}
Status: {row['status']}
Attributes: {row['attributes']}
Description: {row['description']}
"""
    documents.append(Document(page_content=content))
print("ingestion completed")

# ==============================
# ✂️ SPLIT
# ==============================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = splitter.split_documents(documents)
print("chunks completed")

# ==============================
# 🔎 VECTOR DATABASE
# ==============================


db = Chroma.from_documents(documents, GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=google_api_key))
print("db ready")

# ==============================
# 🤖 RAG FUNCTION
# ==============================

SYSTEM_PROMPT = """
You are a helpful supermarket assistant.
Answer ONLY using the given context.
If answer not found, say you don't have that information.
"""

def ask_question(user_query):

    retrieved_docs = db.similarity_search(user_query, k=3)

    context = ""
    for doc in retrieved_docs:
        context += doc.page_content + "\n\n"

    final_prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

User Question:
{user_query}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=final_prompt
    )

    return response.text
print(" response.text")

# ==============================
# 🚀 FASTAPI APP
# ==============================

app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Backend is running 🚀"}

@app.post("/ask")
def ask(query: Query):
    return {"answer": f"You asked: {query.question}"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)