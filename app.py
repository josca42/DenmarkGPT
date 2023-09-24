import streamlit as st
import pandas as pd
from query import get_table
import pickle
from map_viz import plot_map
from plot import create_px_plot
from network import (
    create_table_tree,
    create_sourounding_tables_tree,
    create_dst_tables_tree,
)
from query import (
    find_table,
    decide_table_specs,
    get_table,
    postprocess_table,
)
from actions import match_action, explore_dst_data
from data import AVATARS, KOMMUNER_ID, REGIONER_ID, ALL_GEO_IDS
from app_utils import apply_filters, create_filter_boxes
import streamlit_antd_components as sac

st.set_page_config(layout="wide")
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
if "previous_prompt" not in st.session_state:
    st.session_state.previous_prompt = None
    st.session_state.df = None
    st.session_state.metadata_df = None
    st.session_state.action = -1
    st.session_state.table_ids = None
    st.session_state.tree_table_ids = None
if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.sidebar:
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            st.markdown(
                "Hi, I am your research assistant☺️ I am here to help you with any question you might have, where some of Denmarks statistics data can be helpful☺️"
            )


# Add chat input in main app
if prompt := st.chat_input("Your wish is my command"):
    with st.sidebar:
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(prompt)

# Specify main layout
col1, col2 = st.columns([0.7, 0.3])


###   Determine state   ###
if prompt != st.session_state.previous_prompt and prompt is not None:
    st.session_state.previous_prompt = prompt

    action = match_action(prompt)
    if action == 1:
        df, metadata_df, response_txt = get_table(prompt, st=st)
    elif action == 2:
        response_txt, table_ids = explore_dst_data(prompt, st=st)
        df, metadata_df = None, None
        st.session_state.table_ids = table_ids
    else:
        df, metadata_df, response_txt = get_table(prompt, st=st)

    st.session_state.df = df.copy() if df is not None else None
    st.session_state.metadata_df = metadata_df
    st.session_state.new_prompt = True
    st.session_state.action = action

else:
    df = st.session_state.df.copy() if st.session_state.df is not None else None
    metadata_df = st.session_state.metadata_df if st.session_state.metadata_df else {}
    st.session_state.new_prompt = False
    response_txt = None


if st.session_state.action == 2:
    with col1:
        tree_table_ids, form_button = create_dst_tables_tree(
            st.session_state.table_ids, st=st
        )
        st.session_state.tree_table_ids = tree_table_ids
        if form_button:
            st.info(
                f"""{len(tree_table_ids)} tables have been selected. When answering questions only these table will be considered. To use all tables click the "remove table selection" button""",
                icon="ℹ️",
            )

if st.session_state.tree_table_ids:
    with col2:
        rt_button = st.button("Remove table selection", type="primary")

    if rt_button:
        st.session_state.tree_table_ids = None
        st.session_state.table_ids = None


if st.session_state.action in [1, 3]:
    with col2:
        with st.expander("Similar tables", False):
            select_table = create_sourounding_tables_tree(
                metadata_df["table_id"], st=st
            )

        with st.expander("Selected table", True):
            _ = create_table_tree(metadata_df["table_info"], metadata_df["specs"])

        with st.expander("Filters", True):
            # tab = sac.tabs(
            #     [sac.TabsItem(label="Include"), sac.TabsItem(label="Exclude")],
            #     format_func="title",
            #     align="center",
            # )
            df, select_geo_type, select_multi, variables = create_filter_boxes(
                df, metadata_df, st
            )

    with col1:
        df = apply_filters(df, select_multi, metadata_df)

        if "geo" in metadata_df and select_geo_type is not None:
            deck = plot_map(df, metadata_df, select_geo_type)
            if deck is not None:
                st.pydeck_chart(deck)
        else:
            deck = None

        fig, plot_code_str = create_px_plot(
            df=df,
            prompt=prompt,
            metadata_df=metadata_df,
            variables=variables,
            plot_code=None,
            st=st,
        )
        st.plotly_chart(fig, use_container_width=True)

        df_table = df[df.nunique().pipe(lambda s: s[s > 1]).index].copy()
        if "TID" in df.columns:
            df.set_index(df.columns[:-1]).sort_index().unstack("TID").reset_index()
        st.dataframe(df_table, hide_index=True)

        st.session_state.plot_code_str = plot_code_str

with st.sidebar:
    for msg in reversed(st.session_state.messages):
        with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
            st.markdown(msg["content"])

if response_txt:
    st.session_state.messages.append(dict(role="user", content=prompt))
    st.session_state.messages.append(dict(role="assistant", content=response_txt))
