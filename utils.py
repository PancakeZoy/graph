import networkx as nx
import numpy as np

def create_barbell_graph(n):
    # Create a barbell graph with n nodes in each complete graph
    G = nx.barbell_graph(n, 0)
    
    # Add 'label' attribute for each node
    for node in G.nodes():
        if node < n:
            G.nodes[node]['label'] = 0
        else:
            G.nodes[node]['label'] = 1
    return G


def NodeLabel_to_communities(pred_label, node_order):
    label_assign = {}
    for node, label in enumerate(pred_label):
        if label in label_assign:
            label_assign[label].add(int(node_order[node]))
        else:
            label_assign[label] = {int(node_order[node])}
    return list(label_assign.values())


def community_to_NodeLabel(node_list, comm):
    node_to_cluster = {}
    for cluster_id, cluster in enumerate(comm):
        for node in cluster:
            node_to_cluster[node] = cluster_id
    NodeLabel = np.array([node_to_cluster[node] for node in node_list]).astype(int)
    return NodeLabel
