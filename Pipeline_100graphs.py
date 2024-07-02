from model import GraphEmbd
import networkx as nx
from utils import NodeLabel_to_communities, community_to_NodeLabel
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time
from math import ceil
from tqdm import tqdm

prod_result = pd.read_csv('SampleGraph/ind_CoreHH.csv')
prod_result.head()

edges = pd.read_csv('SampleGraph/edge_list.csv')
edges.columns = ['source', 'target', 'weight', 'hhcluster_id']
edges.head()

HHC_size = pd.read_csv('SampleGraph/HH_size.csv')
random.seed(35)
# HHC_id_set = random.sample(list(HHC_size[HHC_size.h_size>20].hhcluster_id), 100)
HHC_id_set = random.sample(list(HHC_size.hhcluster_id), 10000)

def embd_clust(HHC_id):
    edges_sub = edges[edges.hhcluster_id==HHC_id]
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
    prod_sub = prod_result[prod_result.hhcluster_id == HHC_id]
    if len(np.setxor1d(model.nodes,prod_sub.ind_id))>0:
        raise ValueError('Mismatched nodes found between edges list and production result')
    node_comm = NodeLabel_to_communities(list(prod_sub.core_hh_id), list(prod_sub.ind_id))
    model.node_label['Prod'] = community_to_NodeLabel(node_list, node_comm)
    model.modularity('Prod')
    model.triangles('Prod')
    
    # Add run time to model metric
    model.metric['RunTime'] = {'KMeans': KM_time, 'Louvian': louvian_time}
    return model.metric

result = {}
for HHC_id in tqdm(HHC_id_set, desc="Status", leave=True, unit="graph"):
    result[HHC_id] = embd_clust(HHC_id)
    
mod_kmeans = [metric['modularity']['KMeans'] for metric in result.values()]
mod_louvian = [metric['modularity']['Louvian'] for metric in result.values()]
mod_prod = [metric['modularity']['Prod'] for metric in result.values()]
mod_df= pd.DataFrame({'KMeans': mod_kmeans, 'Louvian': mod_louvian, 'Production': mod_prod})
mod_melt = mod_df.melt(var_name='Methods', value_name='Modularity')
sns.boxplot(x='Methods', y='Modularity', data=mod_melt, palette=['blue', 'red', 'orange'], showfliers=False)
plt.show()

tri_kmeans = [metric['triangles']['KMeans'] for metric in result.values()]
tri_louvian = [metric['triangles']['Louvian'] for metric in result.values()]
tri_prod = [metric['triangles']['Prod'] for metric in result.values()]
tri_df= pd.DataFrame({'KMeans': tri_kmeans, 'Louvian': tri_louvian, 'Production': tri_prod})
tri_melt = tri_df.melt(var_name='Methods', value_name='triangles')
sns.boxplot(x='Methods', y='triangles', data=tri_melt, palette=['blue', 'red', 'orange'], showfliers=False)
plt.show()

time_kmeans = [metric['RunTime']['KMeans'] for metric in result.values()]
time_louvian = [metric['RunTime']['Louvian'] for metric in result.values()]
print(f'The average runtime for KMeans vs Louvian is {np.mean(time_kmeans):.4f} vs {np.mean(time_louvian):.4f} seconds')
print(f'The median runtime for KMeans vs Louvian is {np.median(time_kmeans):.4f} vs {np.median(time_louvian):.4f} seconds')
