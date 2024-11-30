import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.local_llm import load_local_llm, ask_llm

model, tokenizer = load_local_llm()

st.title("RH ChatBot")
query = st.text_input("Enter your prompt:")

if query:
    try:
        response = ask_llm(model, tokenizer, query)
        st.subheader("Response:")
        st.write(response)
    except Exception as e:
        st.error(f"Error generating response: {e}")