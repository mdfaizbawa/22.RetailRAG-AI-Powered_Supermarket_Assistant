import streamlit as st
import requests
import uuid

st.set_page_config(
    page_title="🛒 RetailRAG – AI-Powered Supermarket Assistant",
    layout="wide"
)

# -------------------------
# 🎨 PREMIUM STYLING
# -------------------------
st.markdown("""
<style>

body {
    background-color: #f4f6f9;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 10px;
}

.chat-container {
    max-height: 65vh;
    overflow-y: auto;
    padding-bottom: 80px;
}

.side-img img {
    border-radius: 25px;
    box-shadow: 0px 15px 35px rgba(0,0,0,0.15);
}

.bottom-input {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 50%;
    z-index: 999;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# 🧠 SESSION STATE
# -------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# 🏗 LAYOUT
# -------------------------
left, center, right = st.columns([1.3, 2, 1.3])

# -------------------------
# 🍍 LEFT IMAGE (Your Fruits)
# -------------------------
with left:
    st.markdown('<div class="side-img">', unsafe_allow_html=True)
    st.image("fruits.png", use_container_width=True)  # <-- Save your attached image as fruits.png
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# 💬 CENTER CHAT
# -------------------------
with center:

    st.markdown('<div class="main-title">🛒 RetailRAG – AI-Powered Supermarket Assistant</div>', unsafe_allow_html=True)

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# 🌶 RIGHT IMAGE (Maximized)
# -------------------------
with right:
    st.markdown('<div class="side-img">', unsafe_allow_html=True)
    st.image(
        "https://images.unsplash.com/photo-1596040033229-a9821ebd058d",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# ⬇ FIXED BOTTOM INPUT
# -------------------------
st.markdown('<div class="bottom-input">', unsafe_allow_html=True)

user_input = st.chat_input("Ask about products...")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# 🔁 HANDLE MESSAGE
# -------------------------
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={
                "session_id": st.session_state.session_id,
                "message": user_input
            },
            timeout=60
        )
        reply = response.json()["reply"]

    except:
        reply = "⚠️ Backend not responding."

    st.session_state.messages.append({"role": "assistant", "content": reply})

    st.rerun()