import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from huggingface_hub import InferenceClient


# API KEY

HF_TOKEN = "****"

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)

print("API key loaded")

app = FastAPI()

#SESSION MEMORY STORE
chat_sessions = {}

#LOAD DATA

df = pd.read_csv("supermarket_rag_dataset.csv")

documents = []

for _, row in df.iterrows():
    content = f"""
Product: {row['title']}
Category: {row['category']}
Brand: {row['brand']}
Price: {row['price_inr']} INR
Stock: {row['stock']}
Description: {row['description']}
"""
    documents.append(Document(page_content=content))

print("ingestion completed")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = splitter.split_documents(documents)
print("chunks completed")

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN
)

db = Chroma.from_documents(documents, embeddings)
print("db ready")


#CHAT FUNCTION (STATEFUL)

SYSTEM_PROMPT ="""
You are RetailRAG, a helpful and professional AI-powered supermarket assistant.

Your responsibilities:

1. Answer ONLY using the provided context.
2. If the answer is not found in the context, respond exactly with:
   "I'm unable to provide the information you're looking for."
3. Do NOT use outside knowledge.
4. Assistant responses must always be in English.
5. If the user writes in another language, always respond in English.
6. First determine whether the user is asking about a specific product or multiple products.
7. If the customer makes incorrect assumptions about a product (price, availability, brand, etc.),
   politely correct them using only the provided context.
8. Ensure every fact you mention is strictly supported by the given product data.
9. Validate internally that the response fully answers the customer's question.
10. If the retrieved context is insufficient to provide a complete and accurate answer,
    respond exactly with:
    "I'm unable to provide the information you're looking for."

Response Rules:
- Be clear and professional.
- Be concise but informative.
- Format product details neatly (Price, Stock, Description when relevant).
- Do not fabricate or assume missing information.
"""


def chat_with_memory(session_id, user_input):

    if session_id not in chat_sessions:
        chat_sessions[session_id] = SYSTEM_PROMPT.strip()

    # Retrieve RAG context
    retrieved_docs = db.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Add to conversation history
    chat_sessions[session_id] += f"\nUser: {user_input}\nContext: {context}\nAssistant:"

    response = client.chat_completion(
        messages=[
            {"role": "user", "content": chat_sessions[session_id]}
        ],
        max_tokens=300,
        temperature=0.3
    )

    reply = response.choices[0].message.content

    chat_sessions[session_id] += f" {reply}\n"

    return reply

#FASTAPI ROUTES

class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    reply = chat_with_memory(request.session_id, request.message)
    return {"reply": reply}