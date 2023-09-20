import streamlit as st
import pandas as pd
from query import get_table
import pickle
from map_viz import plot_map
from plot import create_px_plot
from network import create_table_tree, create_dst_tables_tree
from query import find_table, decide_table_specs, get_table

st.set_page_config(layout="wide")
# st.markdown(
#     """
#         <style>
#                .block-container {
#                     padding-top: 1rem;
#                     padding-bottom: 0rem;
#                     padding-left: 5rem;
#                     padding-right: 5rem;
#                 }
#         </style>
#         """,
#     unsafe_allow_html=True,
# )

st.markdown(
    f"""
    <style>
    .stApp .main .block-container{{
        padding-top:30px
    }}
    .stApp [data-testid='stSidebar']>div:nth-child(1)>div:nth-child(2){{
        padding-top:50px
    }}
    iframe{{
        display:block;
    }}
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
    with st.sidebar:
        st.write(prompt)


col1, col2 = st.columns([0.75, 0.25])

### Get the table ###
if prompt != st.session_state.previous_prompt:
    table_id = find_table(prompt, k=5, rerank=True)

    with col2:
        with st.expander("Similar tables", False):
            create_dst_tables_tree(table_id)

    table_metadata = decide_table_specs(prompt, table_id, st=st)

    with col2:
        with st.expander("Table used", True):
            create_table_tree(table_metadata["table_info"], table_metadata["specs"])

    table_df = get_table(table_id, table_metadata["specs"])

    with col2:
        with st.expander("Filters", True):
            filters = {}
            variables = []
            for var, select in metadata_df["specs"].items():
                if select == ["*"]:
                    values = [v["text"] for v in metadata_df[var]["values"]]
                else:
                    values = select

                if len(values) > 1:
                    st.multiselect(var[0].upper() + var[1:].lower(), values)

                variables.append({"name": var, "text": metadata_df[var]["text"]})

    with col1:
        deck = plot_map(df, metadata_df)
        st.pydeck_chart(deck)

        fig = create_px_plot(
            df=df,
            question=prompt,
            data_descr=metadata_df["description"],
            variables=variables,
        )
        st.plotly_chart(fig, use_container_width=True)

with st.sidebar:
    for message in reversed(st.session_state.messages):
        with st.chat_message("assistant", avatar="👨‍🏫"):
            st.markdown(message["content"])
