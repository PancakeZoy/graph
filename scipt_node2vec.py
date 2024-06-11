from model import GraphEmbd
from utils import *

G = create_barbell_graph(5)
model = GraphEmbd(G)

model.embd_init()
model.fit()
model.umap()
model.plot_embd_umap(method='True', title='UMAP of node embedding (True labels)')

model.k_selection((2,6))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
model.plot_trace_K('modularity', ax=ax1, c='red')
model.plot_trace_K('wss', ax=ax2)
model.plot_trace_K('silhouette', ax=ax3)
plt.show()

model.kmeans(n_clusters=2)
model.louvian(seed=3)

model.draw_graph(with_labels=True, method='KMeans')

