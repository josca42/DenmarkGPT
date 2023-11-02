import streamlit_antd_components as sac
import networkx as nx
import pickle
from jinja2 import Template
from dst.db import crud
from dst.data import KOMMUNER_ID, REGIONER_ID, ALL_GEO_IDS, table2node, G


def write_table_selected_update(table, lang, st):
    if lang == "en":
        TABLE_SELECTED = Template(
            """The following table has been selected: {{ table_id }} - {{ table_descr }}"""
        )
    else:
        TABLE_SELECTED = Template(
            """Følgende table er valgt: {{ table_id }} - {{ table_descr }}"""
        )
    with st.sidebar:
        table_msg = TABLE_SELECTED.render(
            table_id=table["id"],
            table_descr=table["description"],
        )
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(table_msg)

    return table_msg


def write_table_large_update(n_obs, lang, st):
    if lang == "en":
        TABLE_LARGE_MSG = Template(
            """The table has {{ n_obs }} observations. Getting data from DST can take a bit of time."""
        )
    else:
        TABLE_LARGE_MSG = Template(
            """Tabellen der hentes fra DST har {{ n_obs }} rækker. Det kan tage lidt tid at hente tabellen."""
        )
    st.toast(
        TABLE_LARGE_MSG.render(n_obs=str(n_obs)),
        icon="ℹ️",
    )


def create_dst_explorer(st):
    with col1:
        tree_table_ids, form_button = create_dst_tables_tree(
            st.session_state.table_ids, lang=lang, st=st
        )
        st.session_state.tree_table_ids = tree_table_ids
        if form_button:
            st.info(
                f"""{len(tree_table_ids)} tables have been selected. When answering questions only these table will be considered. To use all tables click the "remove table selection" button""",
                icon="ℹ️",
            )


def apply_filters(df, select_multi, metadata_df):
    # Apply filters selected in multi select boxes
    vars_processed = ["y"]
    for var, values in select_multi.items():
        if values:
            if df[var].dtype != "object":
                df = df[df[var].astype(str).isin(values)]
            else:
                df = df[df[var].isin(values)]
            vars_processed.append(var)

    # Remove totals unless only totals are in data
    for var in df.columns:
        if var not in vars_processed and var != "geo_id":
            totals = metadata_df[var]["totals"]
            if totals:
                n_unique = len(df[var].unique())
                if n_unique > 1:
                    df = df[~df[var].isin(totals)]

    return df


def create_filter_boxes(df, metadata_df, st):
    if "geo" in metadata_df and df[[metadata_df["geo"]["var"]]].nunique().squeeze() > 1:
        select_geo_type = st.selectbox(
            "Region type", metadata_df["geo"]["geo_types"], 0
        )
        text2id = metadata_df["geo"]["id2text_mapping"]
        df["geo_id"] = df[metadata_df["geo"]["var"]].map(text2id)
    else:
        select_geo_type = None

    select_multi = {}
    filters = {}
    variables = []
    for var, select in metadata_df["specs"].items():
        if (
            "geo" in metadata_df
            and var == metadata_df["geo"]["var"]
            and select_geo_type in ["kommuner", "regioner"]
        ):
            if df["geo_id"].isin(ALL_GEO_IDS).any():
                geo_ids = KOMMUNER_ID if select_geo_type == "kommuner" else REGIONER_ID
                values = [
                    v["text"] for v in metadata_df[var]["values"] if v["id"] in geo_ids
                ]
                df = df[df[var].isin(values)]
            else:
                values = []
        elif select == ["*"]:
            values = [v["text"] for v in metadata_df[var]["values"]]
        else:
            val_id2text = {val["id"]: val["text"] for val in metadata_df[var]["values"]}
            values = [val_id2text[s] for s in select]

        if len(values) > 1:
            select_multi[var] = st.multiselect(var[0].upper() + var[1:].lower(), values)

        variables.append({"name": var, "text": metadata_df[var]["text"]})

    return df, select_geo_type, select_multi, variables


def create_table_tree(table_info, specs):
    sac_vars = []
    tag_chosen = sac.Tag("chosen", color="blue", bordered=False)
    tag_filtered = sac.Tag("filtered", color="green", bordered=False)
    for var in table_info["variables"]:
        val_selected = specs[var["id"]]
        if val_selected != ["*"]:
            val_selected = [
                val["text"] for val in var["values"] if val["id"] in val_selected
            ]
        else:
            val_selected = []

        sac_vals = (
            [sac.TreeItem(val_s, tag=tag_chosen) for val_s in val_selected]
            if val_selected
            else []
        )
        sac_vals += [
            sac.TreeItem(val["text"])
            for val in var["values"][:10]
            if val["text"] not in val_selected
        ]
        if len(var["values"]) > 11:
            sac_vals.append(sac.TreeItem("..."))

        var_tree = sac.TreeItem(
            var["id"],
            tooltip=var["text"],
            tag=tag_filtered if val_selected else None,
            children=sac_vals,
        )
        sac_vars.append(var_tree)

    table_tree = sac.tree(
        items=[
            sac.TreeItem(
                table_info["id"],
                icon="table",
                tooltip=table_info["description"],
                children=sac_vars,
            ),
        ],
        show_line=True,
        open_index=[0, 1],
    )
    return table_tree


def create_dst_tables_tree(table_ids, lang, st):
    st.subheader("All available tables in Denmark Statistics API")
    st.markdown(
        """Below tree graph shows all available tables from Denmarks Statistics API. The tables are grouped into categories. You can select specific tables or categories that you want to limit your search to. Once you have clicked the button "Use selected tables" then all subsequent questions will only consider the selected tables. That is until you click the button "remove table selection". Some of the tables with the most semantically similar table descriptions have been highlighted."""
    )
    crud_table = crud.table_en if lang == "en" else crud.table_da
    df_table = crud_table.get_table().set_index("id").sort_index()

    def has_table_as_successor(node, table_ids):
        if node == "0":
            return False

        if set(G.nodes[node]["tables"]).intersection(set(table_ids)):
            return True

        successors = get_all_successors(G, node)
        for successor in successors:
            if set(G.nodes[successor]["tables"]).intersection(set(table_ids)):
                return True
        return False

    def create_tree(node):
        children = list(G.successors(node))
        sac_children = []
        for child in children:
            sac_child = create_tree(child)
            if sac_child:
                sac_children.append(sac_child)

        sac_tables = [
            sac.TreeItem(
                df_table.loc[t_id, "description"],
                icon="table",
                tag=sac.Tag("Table candidate", color="blue", bordered=False),
            )
            if t_id in table_ids
            else sac.TreeItem(df_table.loc[t_id, "description"], icon="table")
            for t_id in G.nodes[node]["tables"]
            if t_id in set(df_table.index)
        ]

        tag_subj = (
            sac.Tag("Subject with table candidates", color="green", bordered=False)
            if has_table_as_successor(node, table_ids)
            else None
        )

        return (
            sac.TreeItem(
                G.nodes[node]["description"],
                tag=tag_subj,
                children=sac_tables + sac_children,
            )
            if sac_tables or sac_children
            else None
        )

    with st.form("Similar tables"):
        root_tree = create_tree("0")
        table_tree = sac.tree(
            items=[root_tree],
            show_line=True,
            open_index=[0, 1],
            checkbox=True,
        )
        form_button = st.form_submit_button("Use selected tables")

    table_ids = (
        df_table[df_table["description"].isin(table_tree)].index.to_list()
        if table_tree
        else None
    )
    return table_ids, form_button


def get_all_successors(G, node):
    successors = set()
    nodes_to_check = [node]

    while nodes_to_check:
        current_node = nodes_to_check.pop()
        current_successors = list(G.successors(current_node))
        nodes_to_check.extend(current_successors)
        successors.update(current_successors)

    return successors


def get_correct_update_specs_from_metadata(table_metadata):
    specs = table_metadata["specs"]
    var_many_ids = [v["id"] for v in table_metadata["var_many"]]
    for var, vals in specs.items():
        if var in var_many_ids and not (vals == ["*"] or vals == ["latest"]):
            val_id2text = {
                val["id"]: val["text"] for val in table_metadata[var]["values"]
            }
            values = [val_id2text[s] for s in vals]
            specs[var] = values
    return specs


def intro_page(st):
    st.header("Denmark statistics GPT", divider="rainbow")
    st.markdown(
        """Meet your personal GPT research assistant that can access all the publicly available data at Denmarks Statistics😄. Whatever questions you have about Danish society. Ask away!.

A 5 minute introduction and demo of the program can be viewed [here](https://youtu.be/pBPuyM_DMk4). For the best experience see the quick demo first😉.

In the upper left corner you can choose between English and Danish. The quality of the answers is sometimes better in english. Mostly due to out of the box support for english being better. When time allows I'll fix this.

This is a quick prototype developed in my sparetime. So expect weird bugs and unintented behavior to happen from time to time. Also due to rate limits on Denmark Statistics and OpenAI's API then the response time can be slow if a large number of users are using the app at the same time. If you experience this then please try again later😉."""
    )
    st.video("https://youtu.be/pBPuyM_DMk4", format="youtube")

    st.subheader("Use of command line")
    st.markdown(
        """The main form of interaction is by writing questions/commands. Your can write 3 different forms of requests.

 1) Specific questions such as "how has the unemployment evolved".
 2) Updates or changes to an existing question such as "include both sexes".
 3) Exploratory requests about the tables or information available at Denmark Statistics. An example could be: "How could I go about looking into unemployment".

When asking a specific question then GPT will find an appropriate table and device the right query to send to Denmark Statistics API. GPT will then provide and overview of the table, plot the data and add the possibility to filter the data. In order to limit the demand on Denmark Statistics API then GPT will try to request aggregate numbers for the variables where a question does not imply that this variable is important. As an example the question "How has unemployment evolved" will typically specify to Denmarks Statistics API to get the aggregate number for Men and Women since the distinction sex is not important to the question. If your are interested in differences depending on sex then you can either ask the question "How has unemployment evolved over time and for different sexes" or simply ask GPT to update the existing request by writing "include both sexes".

For more control of, which tables GPT should use then ask a more exploratory question and then GPT will give you an overview of the different tables available in Denmark Statistics. You can then select specific tables or categories that you wanna limit your search to. Once you have clicked the button "Use selected tables" then all subsequent question will only consider the selected tables. That is until you click the button "removed table selection"."""
    )
    st.subheader("Layout")
    st.markdown(
        """The sidebar is where the GPT messages are streamed and whenever a question is asked then the center of the app is where the plots will appear. To the right will be filter menus for filtering the data retreived from the Denmark Statistics API, a table overview and an expandable box showing similar available tables."""
    )
    st.subheader("Closing remarks")
    st.markdown(
        """This is a quick demo of how one can connect GPT with Denmarks Statistics excellent [public API](https://www.dst.dk/da/Statistik/brug-statistikken/muligheder-i-statistikbanken/api).

Everything about this demo can be made better, faster and cheaper.

And remember this is the worst this technology will ever be. And it is only just getting started😮."""
    )
    st.subheader("Contact")
    st.markdown("Feel free to contact me at jonathanscharff@gmail.com")
