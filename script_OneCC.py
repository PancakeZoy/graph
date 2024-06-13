from model import GraphEmbd
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
nx.draw(G, node_size=5)

concom = sorted(list(nx.connected_components(G)), key=len, reverse=True)
concom_size = np.array([len(cc) for cc in concom])

sub_G = G.subgraph(concom[0])
model = GraphEmbd(sub_G)
model.draw_graph(with_labels=False, node_size=5)

# Fit the node2vec model
model.embd_init(seed=4)
model.fit()        

# Run PCA on the node2vec embeddings
model.pca()
model.plot_embd(reduction='PCA', title='PCA plot of node embedding', with_labels=False)

#Perform UMAP on the node2vec embeddings, for visualization and K-Means clustering
model.umap(n_jobs=1, reduction='PCA')
model.plot_embd( title='UMAP of node embedding', with_labels=False)

# Perform model selection to choose the number of clusters K
model.hyper_tune(grid=range(4,15), method = 'KMeans')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
model.plot_trace(method='KMeans', score='modularity', ax=ax1, c='red')
model.plot_trace(method='KMeans', score='silhouette', ax=ax2)
plt.show()

# Therefore we select K = 7
k_optim = model.grid['KMeans'][np.argmax(model.trace['KMeans']['modularity'])]
model.kmeans(n_clusters=k_optim)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
model.draw_graph(method='KMeans', with_labels=False, ax=ax1)
model.plot_embd(method='KMeans', title='UMAP of node embedding (KMeans labels)', with_labels=False, ax=ax2)
plt.show()

# Perform model selection to choose min_cluster_size of HDBSCAN
model.HDBSCAN()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
model.draw_graph(method='HDBSCAN', with_labels=False, ax=ax1)
model.plot_embd(method='HDBSCAN', title='UMAP of node embedding (HDBSCAN labels)', with_labels=False, ax=ax2)
plt.show()

# Now we run Louvian's method with default settings on the graph data, and plot the result out colored by the corresponding cluster labels
model.louvian(seed=3)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
model.draw_graph(method='Louvian', with_labels=False, ax=ax1)
model.plot_embd(method='Louvian', title='UMAP of node embedding (Louvian labels)', with_labels=False, ax=ax2)
plt.show()

# Let's compare them all together, will modurality calculated.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
model.draw_graph(method='KMeans', with_labels=False, 
                 title=f"Karate Graph with KMeans labels (Modularity {round(model.metric['modularity']['KMeans'],4)})", ax=ax1)
model.draw_graph(method='HDBSCAN', with_labels=False, 
                 title=f"Karate Graph with HDBSCAN labels (Modularity {round(model.metric['modularity']['HDBSCAN'],4)})", ax=ax2)
model.draw_graph(method='Louvian', with_labels=False, 
                 title=f"Karate Graph with Louvian labels (Modularity {round(model.metric['modularity']['Louvian'],4)})", ax=ax3)
plt.show()

