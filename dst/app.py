import streamlit as st
import pandas as pd
import pickle
from dst import plot
from dst.query import (
    find_table_candidates,
    create_api_call,
    get_table_from_api,
    determine_query_type,
)
from dst.data import AVATARS, KOMMUNER_ID, REGIONER_ID, ALL_GEO_IDS
from dst.app_funcs import (
    apply_filters,
    create_filter_boxes,
    write_table_large_update,
    write_table_selected_update,
    intro_page,
    create_table_tree,
    create_dst_tables_tree,
    convert_spec_ids_to_text,
)
import streamlit_antd_components as sac
import json

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
    section[data-testid="stSidebar"] {{
            width: 550px !important; # Set the width to your desired value
    }}
	</style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    lang = st.radio(
        "label", ["EN 🇺🇸", "DA 🇩🇰"], horizontal=True, label_visibility="hidden"
    )
    lang = "en" if "EN" in lang else "da"

if "previous_prompt" not in st.session_state:
    intro_page(st)

# Add initial state variables
if "previous_prompt" not in st.session_state:
    st.session_state.previous_prompt = None
    st.session_state.df = None
    st.session_state.table_metadata = None
    st.session_state.query_type = -1
    st.session_state.table_id = None
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

###   Update state   ###
if prompt != st.session_state.previous_prompt and prompt is not None:
    st.session_state.previous_prompt = prompt
    if st.session_state.table_metadata is not None:
        prev_table_descr = st.session_state.table_metadata["table_info"]["description"]
        prev_api_request = st.session_state.table_metadata["specs"]
        prev_table_id = st.session_state.table_metadata["table_id"]
    else:
        prev_table_descr = ""
        prev_api_request = ""
        prev_table_id = ""
    setting_info = {
        "prompt": prompt,
        "query_type": -1,
        "prev_table_descr": prev_table_descr,
        "prev_api_request": prev_api_request,
        "lang": lang,
    }

    query_type, query_table_descr = determine_query_type(prompt, setting_info)
    setting_info["query_type"] = query_type

    if query_type == 1:
        # Check if a subset of tables have been selected by the user
        # If so, only consider these tables.
        tree_table_ids = (
            st.session_state.tree_table_ids
            if st.session_state.tree_table_ids is not None
            else None
        )

        table_selected, table_candidates = find_table_candidates(
            table_descr=query_table_descr,
            lang=lang,
            subset_table_ids=tree_table_ids,
            k=10,
            query=prompt,
            rerank=True,
        )
        setting_info["table_id"] = table_selected["id"]
        table_msg = write_table_selected_update(table=table_selected, lang=lang, st=st)
        api_call, table_metadata, api_call_txt = create_api_call(
            query=prompt,
            table_id=table_selected["id"],
            setting_info=setting_info,
            update_request="",
            st=st,
        )
        if api_call is None:
            df, table_metadata = None, None
        else:
            if table_metadata["n_obs"] > 10_000:
                write_table_large_update(
                    n_obs=table_metadata["n_obs"], lang=lang, st=st
                )
            df = get_table_from_api(
                table_id=table_selected["id"], api_call=api_call, lang=lang
            )

    # Update request to existing table
    elif query_type == 2:
        table_id = st.session_state.table_id
        setting_info["table_id"] = table_id
        update_request = json.dumps(
            st.session_state.table_metadata["specs"], ensure_ascii=False
        )
        api_call, table_metadata, api_call_txt = create_api_call(
            query=prompt,
            table_id=table_id,
            setting_info=setting_info,
            update_request=update_request,
            st=st,
        )
        df = get_table_from_api(table_id=table_id, api_call=api_call, lang=lang)
        table_msg = None
        table_candidates = None

    else:
        pass

    st.session_state.df = df.copy() if df is not None else None
    st.session_state.table_metadata = table_metadata
    st.session_state.new_prompt = True
    st.session_state.query_type = query_type
    st.session_state.table_id = table_metadata["table_id"] if table_metadata else None
    st.session_state.setting_info = setting_info

else:
    df = st.session_state.df.copy() if st.session_state.df is not None else None
    metadata_df = (
        st.session_state.table_metadata if st.session_state.table_metadata else None
    )
    st.session_state.new_prompt = False
    api_call_txt = None
    setting_info = st.session_state.setting_info
    table_msg = None

###   Update layout   ###
if st.session_state.query_type in [1, 2]:
    if st.session_state.table_metadata is not None:
        metadata_df = st.session_state.table_metadata
        with col2:
            with st.expander("Selected table", True):
                _ = create_table_tree(metadata_df["table_info"], metadata_df["specs"])

            with st.expander("Filters", True):
                df, select_geo_type, select_multi, variables = create_filter_boxes(
                    df, metadata_df, st
                )

        with col1:
            df = apply_filters(df, select_multi, metadata_df)

            if "geo" in metadata_df and select_geo_type is not None:
                deck = plot.map(df, metadata_df, select_geo_type)
                if deck is not None:
                    st.pydeck_chart(deck)

            fig, plot_code_str = plot.px_fig(
                df=df,
                prompt=prompt,
                metadata_df=metadata_df,
                variables=variables,
                setting_info=setting_info,
                st=st,
            )
            st.plotly_chart(fig, use_container_width=True)

            # FIXME: If only one number is returned then the table will be empty
            df_table = df[df.nunique().pipe(lambda s: s[s > 1]).index].copy()
            st.dataframe(df_table, hide_index=True)

            st.session_state.plot_code_str = plot_code_str
    else:
        with col1:
            table_ids = [t["id"] for t in table_candidates]
            tree_table_ids, form_button = create_dst_tables_tree(
                table_ids=table_ids, lang=lang, st=st
            )
            st.session_state.tree_table_ids = tree_table_ids
            if form_button:
                st.info(
                    f"""{len(tree_table_ids)} tables have been selected. When answering questions only these table will be considered. To use all tables click the "remove table selection" button""",
                    icon="ℹ️",
                )
else:
    pass

if st.session_state.tree_table_ids:
    with col2:
        rt_button = st.button("Remove table selection", type="primary")

    if rt_button:
        st.session_state.tree_table_ids = None
        st.session_state.table_ids = None


with st.sidebar:
    for msg in reversed(st.session_state.messages):
        with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
            st.markdown(msg["content"])

if api_call_txt:
    st.session_state.messages.append(dict(role="assistant", content=api_call_txt))
    if table_msg is not None:
        st.session_state.messages.append(dict(role="assistant", content=table_msg))
    st.session_state.messages.append(dict(role="user", content=prompt))
