import pandas as pd
import networkx as nx
import numpy as np

edges = pd.read_csv('SampleGraph/edge_list.csv')
edges.columns = ['source', 'target', 'weight', 'hhcluster_id']
grouped = edges.groupby('hhcluster_id')
def calculate_metrics(group):
    h_id = group.name
    n_nodes = len(np.union1d(group.source, group.target))
    G = nx.from_pandas_edgelist(group)
    n_tri = sum(nx.triangles(G).values()) // 3
    return pd.Series({'hhcluster_id': h_id, 'h_size': n_nodes, 'n_triangles': n_tri})
HHC_size = grouped.apply(calculate_metrics).reset_index(drop=True)
HHC_size.sort_values(by=['h_size','n_triangles'], ascending=[False, False], inplace=True)
HHC_size.to_csv('SampleGraph/HH_size.csv', index=False)
