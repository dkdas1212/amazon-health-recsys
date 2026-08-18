"""
streamlit_chat.py
Chat-style front end for the agentic layer. Run alongside (or instead of)
your existing recsys Streamlit dashboard.

Run: streamlit run streamlit_chat.py
"""

import streamlit as st
from agent import AGENT
from db_logger import log_interaction

st.set_page_config(page_title="Agentic Recsys Assistant", page_icon="🛒")
st.title("🛒 Agentic Shopping Assistant")
st.caption("Natural-language layer on top of the SVD / content / hybrid recommenders.")

if "history" not in st.session_state:
    st.session_state.history = []

user_id = st.sidebar.text_input("User ID", value="123")
show_trace = st.sidebar.checkbox("Show agent reasoning trace", value=False)

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask for recommendations...")
if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = AGENT.invoke({"query": query, "user_id": user_id, "trace": []})
        st.write(result["response"])
        log_interaction(user_id, query, result["response"], result["trace"])
        if show_trace:
            with st.expander("Reasoning trace"):
                for step in result["trace"]:
                    st.write("- " + step)

    st.session_state.history.append({"role": "assistant", "content": result["response"]})