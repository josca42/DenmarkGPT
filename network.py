import streamlit_antd_components as sac
import networkx as nx
from data import df_table, table2node, G, TABLE_INFO_DIR
import pickle
import streamlit as st


def create_dst_tables_tree(table_ids, st):
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
        form_button = st.form_submit_button("Use selected table")

    table_ids = (
        df_table[df_table["description"].isin(table_tree)].index.to_list()
        if table_tree
        else None
    )
    return table_ids, form_button


def create_sourounding_tables_tree(table_id, st):
    node = table2node[table_id]
    predecessors = get_all_predecessors(G, node)
    sourounding_nodes = get_all_successors(G, predecessors[-1])
    with st.form("Similar tables"):
        sac_sor = []
        for node in sourounding_nodes:
            sac_tables = [
                sac.TreeItem(df_table.loc[t_id, "description"], icon="table")
                if t_id != table_id
                else sac.TreeItem(
                    df_table.loc[t_id, "description"],
                    icon="table",
                    tag=sac.Tag("chosen", color="blue", bordered=False),
                )
                for t_id in G.nodes[node]["tables"]
            ]
            if table_id in G.nodes[node]["tables"]:
                sac_sor.append(
                    sac.TreeItem(
                        G.nodes[node]["description"],
                        tag=sac.Tag("chosen", color="blue", bordered=False),
                        children=sac_tables,
                    )
                )
            else:
                sac_sor.append(
                    sac.TreeItem(G.nodes[node]["description"], children=sac_tables)
                )

        table_tree = sac.tree(
            items=[
                sac.TreeItem(
                    G.nodes[predecessors[0]]["description"],
                    children=[
                        sac.TreeItem(
                            G.nodes[predecessors[1]]["description"], children=sac_sor
                        )
                    ],
                )
            ],
            show_line=True,
            open_index=[0, 1],
            checkbox=True,
        )
        st.form_submit_button("Use selected table")

    table_ids = (
        df_table[df_table["description"].isin(table_tree)].index.to_list()
        if table_tree
        else None
    )
    return table_ids


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
            for val in var["values"][:8]
            if val["text"] not in val_selected
        ]
        if len(var["values"]) > 10:
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


def get_all_predecessors(G, node):
    predecessors = set()
    nodes_to_check = [node]

    while nodes_to_check:
        current_node = nodes_to_check.pop()
        current_predecessors = list(G.predecessors(current_node))
        nodes_to_check.extend(current_predecessors)
        predecessors.update(current_predecessors)

    if "0" in predecessors:
        predecessors.remove("0")

    return list(predecessors)


def get_all_successors(G, node):
    successors = set()
    nodes_to_check = [node]

    while nodes_to_check:
        current_node = nodes_to_check.pop()
        current_successors = list(G.successors(current_node))
        nodes_to_check.extend(current_successors)
        successors.update(current_successors)

    return successors


# if __name__ == "__main__":
#     table_id = "NGLK"
#     create_dst_tables_tree(table_id)
#     create_table_tree(table_id, specs)
