from model import GraphEmbd
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

edges = pd.read_csv('SampleGraph/ind_graph_edge_list_snapshot_20240314_sample01.csv', skiprows=3)
edges.columns = ['source', 'target', 'weight', 'HouseHold_id']
edges = edges.iloc[:-2,]
edges['source'] = edges['source'].astype('int64')
edges['target'] = edges['target'].astype('int64')
edges['HouseHold_id'] = edges['HouseHold_id'].astype('int64')

HHC_size = pd.DataFrame(edges.HouseHold_id.value_counts())
HH_id = HHC_size.index[0]

edges_HH = edges[edges.HouseHold_id==HH_id]
G = nx.from_pandas_edgelist(edges_HH, 'source', 'target', edge_attr='weight')

concom = sorted(list(nx.connected_components(G)), key=len, reverse=True)
concom_size = np.array([len(cc) for cc in concom])
label_kmeans = []; label_HDBSCAN = []

for cc, len_ in zip(concom, concom_size):
    if len_ < 10:
        label_kmeans.append(cc)
        label_HDBSCAN.append(cc)
    else:
        sub_G = G.subgraph(cc)
        model = GraphEmbd(sub_G)
        model.draw_graph(with_labels=False, node_size=5)
        # Fit the node2vec model
        model.embd_init(seed=4)
        model.fit()        
        # Run PCA on the node2vec embeddings
        model.pca()                
        # Perform model selection to choose the number of clusters K
        grid_min = max(2, math.ceil(model.n_nodes/10))
        grid_max = int(model.n_nodes/2)+1
        model.hyper_tune(grid=range(grid_min, grid_max), method = 'KMeans')
        # Therefore we select K = 8
        k_optim = model.grid['KMeans'][np.argmax(model.trace['KMeans']['modularity'])]
        model.kmeans(n_clusters=k_optim)
        # Perform model selection to choose min_cluster_size of HDBSCAN
        model.HDBSCAN(allow_single_cluster=True)
        if -1 in model.node_label['HDBSCAN']:
            raise ValueError('Unclustered points returned')
        label_kmeans.extend(NodeLabel_to_communities(model.node_label['KMeans'], model.nodes))
        label_HDBSCAN.extend(NodeLabel_to_communities(model.node_label['HDBSCAN'], model.nodes))

model = GraphEmbd(G)
model.draw_graph(with_labels=False, node_size=5)
node_list = list(model.G.nodes)

model.node_label['KMeans'] = community_to_NodeLabel(node_list, label_kmeans)
model.modularity('KMeans')
model.draw_graph(method='KMeans', with_labels=False, node_size=5)

model.node_label['HDBSCAN'] = community_to_NodeLabel(node_list, label_HDBSCAN)
model.modularity('HDBSCAN')
model.draw_graph(method='HDBSCAN', with_labels=False, node_size=5)

model.louvian()
model.draw_graph(method='Louvian', with_labels=False, node_size=5)



model = GraphEmbd(G)
model.embd_init(seed=4)
model.fit()        
model.pca()
model.plot_embd(reduction='PCA', title='PCA plot of node embedding', with_labels=False)
model.umap(n_jobs=1, reduction='PCA')
model.plot_embd( title='UMAP of node embedding', with_labels=False)

grid_min = max(2, math.ceil(model.n_nodes/10))
grid_max = int(model.n_nodes/2)+1
model.hyper_tune(grid=range(grid_min, grid_max), method = 'KMeans')
k_optim = model.grid['KMeans'][np.argmax(model.trace['KMeans']['modularity'])]
model.kmeans(n_clusters=k_optim)
model.draw_graph(method='KMeans', with_labels=False, node_size=5)
model.plot_embd(method='KMeans', with_labels=False, s=2)

model.HDBSCAN()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
model.draw_graph(method='HDBSCAN', with_labels=False, ax=ax1)
model.plot_embd(method='HDBSCAN', title='UMAP of node embedding (HDBSCAN labels)', with_labels=False, ax=ax2)
plt.show()



