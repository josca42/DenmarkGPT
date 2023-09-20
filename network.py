import streamlit_antd_components as sac
import networkx as nx
from data import df_table, table2node, G, TABLE_INFO_DIR
import pickle


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
        label="Table",
    )
    return table_tree


def create_dst_tables_tree(table_id):
    node = table2node[table_id]
    predecessors = get_all_predecessors(G, node)
    sourounding_nodes = get_all_successors(G, predecessors[-1])
    sac_sor = []
    for node in sourounding_nodes:
        sac_tables = [
            sac.TreeItem(df_table.loc[table_id, "description"], icon="table")
            for table_id in G.nodes[node]["tables"]
        ]
        sac_sor.append(sac.TreeItem(G.nodes[node]["description"], children=sac_tables))

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
        checkbox=True,
        show_line=True,
    )


def get_all_predecessors(G, node):
    predecessors = set()
    nodes_to_check = [node]

    while nodes_to_check:
        current_node = nodes_to_check.pop()
        current_predecessors = list(G.predecessors(current_node))
        nodes_to_check.extend(current_predecessors)
        predecessors.update(current_predecessors)

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
