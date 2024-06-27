from model import GraphEmbd
import networkx as nx
from utils import NodeLabel_to_communities, community_to_NodeLabel
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time

prod_result = pd.read_csv('SampleGraph_new/ind_CoreHH.csv')
prod_result.head()

edges = pd.read_csv('SampleGraph_new/edge_list.csv')
edges.columns = ['source', 'target', 'weight', 'hhcluster_id']
edges.head()

HHC_size = pd.DataFrame(edges.hhcluster_id.value_counts())
random.seed(35)
HHC_id_set = random.sample(list(HHC_size[HHC_size.hhcluster_id>20].index), 100)

def embd_clust(HHC_id):
    edges_sub = edges[edges.hhcluster_id==HHC_id]
    G = nx.from_pandas_edgelist(edges_sub, 'source', 'target', edge_attr='weight')
    
    concom = sorted(list(nx.connected_components(G)), key=len, reverse=True)
    concom_size = np.array([len(cc) for cc in concom])
    label_kmeans = []
    
    KM_start = time.time()
    for cc, size in zip(concom, concom_size):
        if size < 10:
            label_kmeans.append(cc)
        else:
            sub_G = G.subgraph(cc)
            model = GraphEmbd(sub_G)
            # Fit the node2vec model
            model.embd_init(seed=4, dimensions=20, walk_length = 40, quiet=True)
            model.fit()
            # Perform HDBSCAN clustering to set a basline K for KMeans
            model.HDBSCAN(reduction='node2vec')
            K_HDBSCAN = len(np.unique(model.node_label['HDBSCAN']))
            # Perform model selection to choose the number of clusters K
            grid_min = max(2, K_HDBSCAN-2)
            grid_max = min(K_HDBSCAN+2, int(model.n_nodes/2))
            model.hyper_tune(grid=range(grid_min, grid_max+1), method = 'KMeans', reduction='node2vec')
            # Re-run Kmeans using the optimized K
            k_optim = model.grid['KMeans'][np.argmax(model.trace['KMeans']['modularity'])]
            model.kmeans(n_clusters=k_optim, reduction='node2vec')
            label_kmeans.extend(NodeLabel_to_communities(model.node_label['KMeans'], model.nodes))
    KM_time = time.time()-KM_start
    
    model = GraphEmbd(G)
    node_list = list(model.G.nodes)
    
    # K-Means
    model.node_label['KMeans'] = community_to_NodeLabel(node_list, label_kmeans)
    model.modularity('KMeans')
    
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
    
    # Add run time to model metric
    model.metric['RunTime'] = {'KMeans': KM_time, 'Louvian': louvian_time}
    return model.metric

result = {}
for HHC_id in HHC_id_set:
    result[HHC_id] = embd_clust(HHC_id)
    
mod_kmeans = [metric['modularity']['KMeans'] for metric in result.values()]
mod_louvian = [metric['modularity']['Louvian'] for metric in result.values()]
mod_prod = [metric['modularity']['Prod'] for metric in result.values()]
mod_df= pd.DataFrame({'KMeans': mod_kmeans, 'Louvian': mod_louvian, 'Production': mod_prod})
mod_melt = mod_df.melt(var_name='Methods', value_name='Modularity')
sns.boxplot(x='Methods', y='Modularity', data=mod_melt, palette=['blue', 'red', 'orange'])
plt.show()

time_kmeans = [metric['RunTime']['KMeans'] for metric in result.values()]
time_louvian = [metric['RunTime']['Louvian'] for metric in result.values()]
print(f'The average runtime for KMeans vs Louvian is {np.mean(time_kmeans):.4f} vs {np.mean(time_louvian):.4f} seconds')
print(f'The median runtime for KMeans vs Louvian is {np.median(time_kmeans):.4f} vs {np.median(time_louvian):.4f} seconds')