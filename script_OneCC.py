from model import GraphEmbd
from utils import *
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

edges = pd.read_csv('SampleGraph_new/edge_list.csv')
edges.columns = ['source', 'target', 'weight', 'hhcluster_id']
edges.head()

prod = pd.read_csv('SampleGraph_new/ind_CoreHH.csv')
prod.head()

HHC_size = pd.DataFrame(edges.hhcluster_id.value_counts())
HHC_id = HHC_size.index[0]

edges_subset = edges[edges.hhcluster_id==HHC_id]
prod_subset = prod[prod.hhcluster_id==HHC_id][['ind_id', 'core_hh_id']]

# Just in case if there are multiple connected components in the graph. They are supposed to be presented here tho.
# concom = sorted(list(nx.connected_components(G)), key=len, reverse=True)
# concom_size = np.array([len(cc) for cc in concom])
# sub_G = G.subgraph(concom[0])
G = nx.from_pandas_edgelist(edges_subset, 'source', 'target', edge_attr='weight')
model = GraphEmbd(G)

# Add Production labels to the model
prod_dict = dict(prod_subset.values)
prod_label = [prod_dict[node] for node in G.nodes]
model.node_label['Production'] = prod_label
model.modularity('Production')
model.draw_graph(method='Production', with_labels=False, node_size=30)

# Fit the node2vec model
model.embd_init(seed=4)
model.fit()

# Run PCA on the node2vec embeddings
model.pca()
model.plot_embd(reduction='PCA', title='PCA plot of node embedding', with_labels=False, method='Production')

#Perform UMAP on the node2vec embeddings, for visualization and K-Means clustering
model.umap(n_jobs=1, reduction='PCA')
model.plot_embd( title='UMAP of node embedding', with_labels=False, method='Production')

# Perform model selection to choose the number of clusters K
model.hyper_tune(grid=range(4,15), method = 'KMeans')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
model.plot_trace(method='KMeans', score='modularity', ax=ax1, c='red')
model.plot_trace(method='KMeans', score='silhouette', ax=ax2)
plt.show()

# Therefore we select the optimal K
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
