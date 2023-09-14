import requests
import networkx as nx

subjects = requests.get(
    "https://api.statbank.dk/v1/subjects",
    params={"includeTables": True, "recursive": True, "omitInactiveSubjects": True},
).json()


def add_nodes_edges(G, subjects, parent=None):
    for subj in subjects:
        G.add_node(
            subj["id"],
            description=subj["description"],
            active=subj["active"],
            tables=[t["id"] for t in subj["tables"]],
        )
        if parent:
            G.add_edge(parent, subj["id"])
        if subj["hasSubjects"]:
            add_nodes_edges(G, subj["subjects"], parent=subj["id"])


G = nx.DiGraph()
add_nodes_edges(G, subjects)
nx.write_gml(G, "./data/subjects_graph.gml")
