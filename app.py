import streamlit as st
import pandas as pd
from query import get_table

st.set_page_config(layout="wide")
st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

# Add initial state variables
if "metadata" not in st.session_state:
    st.session_state.metadata = {}
if "previous_prompt" not in st.session_state:
    st.session_state.previous_prompt = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        dict(role="assistant", content="Let's get those bookings sorted!")
    ]


# Add chat input in main app
if prompt := st.chat_input("Your wish is my command"):
    st.write(prompt)

    df, metadata = get_table(prompt, st=st)

    if "geo" in metadata:
        