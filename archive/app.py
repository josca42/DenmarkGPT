import streamlit as st
import pandas as pd
from query import get_table
import pickle
from map_viz import plot_map
from plot import create_px_plot

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
    pass


prompt = "Which municipality has the highest deficit?"
# df, metadata_df = get_table(prompt, st=None)
df = pd.read_parquet("dev.parquet")
metadata_df = pickle.load(open("dev.pkl", "rb"))

col1, col2 = st.columns([0.15, 0.85])

with col1:
    with st.expander("Filters", True):
        filters = {}
        variables = []
        for var, select in metadata_df["specs"].items():
            if select == ["*"]:
                values = [v["text"] for v in metadata_df[var]["values"]]
            else:
                values = select
                # filters[var] = [
                #     v["text"]
                #     for v in metadata_df[var]["values"]
                #     for s in select
                #     if s == v["id"]
                # ]

            if len(values) > 1:
                st.multiselect(var[0].upper() + var[1:].lower(), values)

            variables.append({"name": var, "text": metadata_df[var]["text"]})


with col2:
    # with st.container():
    st.dataframe(df)


col3, col4 = st.columns(2)

with col3:
    # deck = plot_map(df, metadata_df)
    # st.pydeck_chart(deck)
    pass

with col4:
    # filters_stmt = "WHERE " + " AND ".join(
    #     [
    #         f"{k} in {tuple(v)}" if len(v) > 1 else f"{k} = '{v[0]}'"
    #         for k, v in filters.items()
    #     ]
    # )
    fig = create_px_plot(
        df=df,
        question=prompt,
        data_descr=metadata_df["description"],
        variables=variables,
    )
    st.plotly_chart(fig, use_container_width=True)
