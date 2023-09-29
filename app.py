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
from intro import intro_page
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
        padding-top:20px
    }}
    iframe{{
        display:block;
    }}
	.stRadio [role=radiogroup]{{
        align-items: center;
        justify-content: center;
    }}
	</style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    lang = st.radio("label", ["EN 🇺🇸", "DA 🇩🇰"],
                    horizontal=True,
                    label_visibility="hidden")
    lang = "en" if "EN" in lang else "da"

if "previous_prompt" not in st.session_state:
    intro_page(st)

# Add initial state variables
if "previous_prompt" not in st.session_state:
    st.session_state.previous_prompt = None
    st.session_state.df = None
    st.session_state.metadata_df = None
    st.session_state.action_type = -1
    st.session_state.table_ids = None
    st.session_state.tree_table_ids = None
    st.session_state.setting_info = None

if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.sidebar:
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            st.markdown(
                "Hi, I am your research assistant😄 I am here to help you with any question, where Denmarks statistics data can be helpful😄"
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

    if st.session_state.metadata_df is not None:
        prev_table_descr = st.session_state.metadata_df["table_info"][
            "description"]
        prev_api_request = st.session_state.metadata_df["specs"]
        prev_table_id = st.session_state.metadata_df["table_id"]
    else:
        prev_table_descr = ""
        prev_api_request = ""
        prev_table_id = ""

    setting_info = {
        "prompt": prompt,
        "action_type": -1,
        "prev_request_table": prev_table_id,
        "prev_request_api": prev_api_request,
        "lang": lang,
        "table_id": "",
    }
    action_type, table_descr = match_action(prompt, prev_table_descr,
                                            prev_api_request, lang,
                                            setting_info)
    setting_info["action_type"] = action_type

    if action_type in [1, 3]:
        table_ids = (st.session_state.table_ids
                     if st.session_state.table_ids is not None else None)
        df, metadata_df, response_txt = get_table(
            query=prompt,
            lang=lang,
            action=action_type,
            table_descr=table_descr,
            subset_table_ids=table_ids,
            st=st,
            setting_info=setting_info,
        )
    elif action_type == 2:
        response_txt, table_ids = explore_dst_data(prompt,
                                                   lang=lang,
                                                   st=st,
                                                   setting_info=setting_info)
        df, metadata_df = None, None
        st.session_state.table_ids = table_ids
    else:
        pass

    st.session_state.df = df.copy() if df is not None else None
    st.session_state.metadata_df = metadata_df
    st.session_state.new_prompt = True
    st.session_state.action_type = action_type
    st.session_state.table_id = metadata_df["table_id"] if metadata_df else None
    st.session_state.setting_info = setting_info

else:
    df = st.session_state.df.copy(
    ) if st.session_state.df is not None else None
    metadata_df = st.session_state.metadata_df if st.session_state.metadata_df else None
    st.session_state.new_prompt = False
    response_txt = None
    setting_info = st.session_state.setting_info

if st.session_state.action_type == 2:
    with col1:
        tree_table_ids, form_button = create_dst_tables_tree(
            st.session_state.table_ids, lang=lang, st=st)
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

if st.session_state.action_type in [
        1, 3
] and st.session_state.metadata_df is not None:
    with col2:
        with st.expander("Similar tables", False):
            select_table = create_sourounding_tables_tree(
                metadata_df["table_id"], lang=lang, st=st)

        with st.expander("Selected table", True):
            _ = create_table_tree(metadata_df["table_info"],
                                  metadata_df["specs"])

        with st.expander("Filters", True):
            df, select_geo_type, select_multi, variables = create_filter_boxes(
                df, metadata_df, st)

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
            setting_info=setting_info,
            st=st,
        )
        st.plotly_chart(fig, use_container_width=True)

        df_table = df[df.nunique().pipe(lambda s: s[s > 1]).index].copy()
        st.dataframe(df_table, hide_index=True)

        st.session_state.plot_code_str = plot_code_str

with st.sidebar:
    for msg in reversed(st.session_state.messages):
        with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
            st.markdown(msg["content"])

if response_txt:
    st.session_state.messages.append(
        dict(role="assistant", content=response_txt))
    st.session_state.messages.append(dict(role="user", content=prompt))
