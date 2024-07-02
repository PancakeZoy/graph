import networkx as nx
import numpy as np
import time
from math import ceil
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from GraphEmbd.model import GraphEmbd
from GraphEmbd.utils import NodeLabel_to_communities, community_to_NodeLabel

def embd_clust(HHC_id, edges_sub, prod_sub):
    G = nx.from_pandas_edgelist(edges_sub, 'source', 'target', edge_attr='weight')
    concom = sorted(list(nx.connected_components(G)), key=len, reverse=True)
    if len(concom) != 1:
        raise ValueError(f'Household {HHC_id} is not a connected graph!')
    
    KM_start = time.time()
    model = GraphEmbd(G)
    node_list = list(model.G.nodes)
    if len(node_list) <= 3:
        model.node_label['KMeans'] = np.zeros(len(node_list), dtype=int)
        model.modularity('KMeans')
        model.triangles('KMeans')
    else:
        # Fit the node2vec model
        model.embd_init(seed=4, dimensions=20, walk_length = 40, quiet=True)
        model.fit()
        # Perform HDBSCAN clustering to set a basline K for KMeans
        model.HDBSCAN(reduction='node2vec')
        K_HDBSCAN = len(np.unique(model.node_label['HDBSCAN']))
        # Perform model selection to choose the number of clusters K
        grid_min = max(ceil(model.n_nodes/10), K_HDBSCAN-3)
        grid_max = min(K_HDBSCAN+3, ceil(model.n_nodes/3))+1
        if grid_max <= grid_min:
            raise ValueError(f'Invalid searching window: min={grid_min}, max={grid_max}')
        model.hyper_tune(grid=range(grid_min, grid_max), method = 'KMeans', reduction='node2vec')
    KM_time = time.time()-KM_start
    
    # Louvian
    louvian_start = time.time()
    model.louvian()
    louvian_time = time.time()-louvian_start
    
    # Input Production Result
    if len(np.setxor1d(model.nodes,prod_sub.ind_id))>0:
        raise ValueError('Mismatched nodes found between edges list and production result')
    node_comm = NodeLabel_to_communities(list(prod_sub.core_hh_id), list(prod_sub.ind_id))
    model.node_label['Prod'] = community_to_NodeLabel(node_list, node_comm)
    model.modularity('Prod')
    model.triangles('Prod')
    
    # Add run time to model metric
    model.metric['RunTime'] = {'KMeans': KM_time, 'Louvian': louvian_time}
    return model.metric