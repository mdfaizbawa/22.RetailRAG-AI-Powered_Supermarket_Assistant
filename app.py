import streamlit as st
import requests

st.title("🛒 Supermarket RAG Chatbot")

user_input = st.text_input("Ask about products:")

if st.button("Ask"):
    if user_input:
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": user_input}
        )

        answer = response.json()["answer"]

        st.subheader("🤖 Answer:")
        st.write(answer)