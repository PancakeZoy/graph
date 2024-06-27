from model import GraphEmbd
from utils import NodeLabel_to_communities, community_to_NodeLabel
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
prod_subset = prod[prod.hhcluster_id == HHC_id]

# Just in case if there are multiple connected components in the graph. They are supposed to be presented here tho.
# concom = sorted(list(nx.connected_components(G)), key=len, reverse=True)
# concom_size = np.array([len(cc) for cc in concom])
# sub_G = G.subgraph(concom[0])

G = nx.from_pandas_edgelist(edges_subset, 'source', 'target', edge_attr='weight')
model = GraphEmbd(G)

# Add Production labels to the model
node_comm = NodeLabel_to_communities(list(prod_subset.core_hh_id), list(prod_subset.ind_id))
model.node_label['Production'] = community_to_NodeLabel(list(model.G.nodes), node_comm)
model.modularity('Production')
model.draw_graph(method='Production', with_labels=False, node_size=30)

# Fit the node2vec model
model.embd_init(seed=4, dimensions=20, walk_length = 40, quiet=True)
model.fit()

# Run PCA on the node2vec embeddings
model.pca()
model.plot_embd(reduction='PCA', title='PCA plot of node embedding', with_labels=False, method='Production')

#Perform UMAP on the node2vec embeddings, for visualization and K-Means clustering
model.umap(n_jobs=1, reduction='PCA')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
model.draw_graph(method='Production', with_labels=False, ax=ax1)
model.plot_embd(method='Production', title='UMAP of node embedding (Production labels)', with_labels=False, ax=ax2)
plt.show()

# # Perform HDBSCAN clustering to set a basline K for KMeans
model.HDBSCAN(reduction='node2vec')
K_HDBSCAN = len(np.unique(model.node_label['HDBSCAN']))

# Perform model selection to choose the number of clusters K
grid_min = max(2, K_HDBSCAN-2)
grid_max = min(K_HDBSCAN+2, int(model.n_nodes/2))
model.hyper_tune(grid=range(grid_min, grid_max+1), method = 'KMeans', reduction='node2vec')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
model.plot_trace(method='KMeans', score='modularity', ax=ax1, c='red')
model.plot_trace(method='KMeans', score='silhouette', ax=ax2)
plt.show()

# Therefore we select the optimal K
k_optim = model.grid['KMeans'][np.argmax(model.trace['KMeans']['modularity'])]
model.kmeans(n_clusters=k_optim, reduction='node2vec')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
model.draw_graph(method='KMeans', with_labels=False, ax=ax1)
model.plot_embd(method='KMeans', title='UMAP of node embedding (KMeans labels)', with_labels=False, ax=ax2)
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
model.draw_graph(method='Production', with_labels=False, 
                 title=f"Karate Graph with Production labels (Modularity {round(model.metric['modularity']['Production'],4)})", ax=ax2)
model.draw_graph(method='Louvian', with_labels=False, 
                 title=f"Karate Graph with Louvian labels (Modularity {round(model.metric['modularity']['Louvian'],4)})", ax=ax3)
plt.show()