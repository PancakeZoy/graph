from model import GraphEmbd
from utils import *
import matplotlib.pyplot as plt
import pandas as pd

# Initialize the Karate Graph
edges = pd.read_csv("Data/Karate_edges.csv")
node_attrs = pd.read_csv("Data/Karate_NodeAttrs.csv").to_dict()['label']
G = nx.from_pandas_edgelist(df=edges)
nx.set_node_attributes(G, node_attrs, name='label')

# Initialize the node2vec model
model = GraphEmbd(G)
model.draw_graph(with_labels=True, method='True', font_color='red')

# Fit the node2vec model
model.embd_init(seed=4)
model.fit()

#Perform UMAP on the returned node embeddings, for visualization and K-Means clustering
model.umap(n_jobs=1)
model.plot_embd_umap(method='True', title='UMAP of node embedding (True labels)')

# Perform model selection to choose the number of clusters K
model.k_selection((2,15))

# Plot out the score(K)
# Here we present three scores: modularity, within-cluster sum of squared (wss) errors, and silhouette score
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
model.plot_trace_K('modularity', ax=ax1, c='red')
model.plot_trace_K('wss', ax=ax2)
model.plot_trace_K('silhouette', ax=ax3)
plt.show()

# Therefore we select K = 4
model.kmeans(n_clusters=4)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
model.draw_graph(with_labels=True, method='KMeans', font_color='red', ax=ax1)
model.plot_embd_umap(method='KMeans', title='UMAP of node embedding (KMeans labels)', ax=ax2)
plt.show()

# Now we run Louvian's method with default settings on the graph data, and plot the result out colored by the corresponding cluster labels
model.louvian(seed=3)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
model.draw_graph(with_labels=True, method='Louvian', font_color='red', ax=ax1)
model.plot_embd_umap(method='Louvian', title='UMAP of node embedding (Louvian labels)', ax=ax2)
plt.show()

# Let's compare them all together, will modurality calculated.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
model.draw_graph(with_labels=True, method='True', font_color='red', 
                 title=f"Karate Graph with True labels (Modularity {round(model.metric['modularity']['True'],4)})", ax=ax1)
model.draw_graph(with_labels=True, method='KMeans', font_color='red', 
                 title=f"Karate Graph with KMeans labels (Modularity {round(model.metric['modularity']['KMeans'],4)})", ax=ax2)
model.draw_graph(with_labels=True, method='Louvian', font_color='red', 
                 title=f"Karate Graph with Louvian labels (Modularity {round(model.metric['modularity']['Louvian'],4)})", ax=ax3)
plt.show()

