from GraphEmbd.model import GraphEmbd
from GraphEmbd.utils import hopkins
import networkx as nx
import itertools
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics.cluster import adjusted_rand_score

prod = pd.read_csv('../Data/ind_CoreHH.csv')
prod.head()

edges = pd.read_csv('../Data/edge_list.csv')
edges.columns = ['source', 'target', 'weight', 'hhcluster_id']
edges.head()

nodes_in_prod=prod.ind_id.unique()
nodes_in_edge = np.unique(list(edges.source.unique()) + list(edges.target.unique()))
print(f'{len(nodes_in_prod)} unique nodes found in the production result file')
print(f'{len(nodes_in_edge)} unique nodes found in the edge list')
print(f'{len(np.intersect1d(nodes_in_prod, nodes_in_edge))} matched nodes found')

HHC_size = pd.read_csv('../Data/HH_size.csv', index_col = 0)
random.seed(42)
HHC_id = random.sample(list(HHC_size[HHC_size.h_size == 16].index), 1)[0]


edges_subset = edges[edges.hhcluster_id==HHC_id]
G = nx.from_pandas_edgelist(edges_subset, 'source', 'target', edge_attr='weight')

prod_subset = prod[prod.hhcluster_id==HHC_id]
prod_nodel_label = list(prod_subset.core_hh_id)
k_hhc = len(np.unique(prod_nodel_label))

dim_range = range(10, 110, 10)
walk_len_range = range(5, 85, 10)
n_walks_range = [5, 10, 15]
# p_range = [0.5,0.75, 1.0]
# q_range = [0.5,0.75, 1.0]
p_range = [0.75]
q_range = [1.0]
window_range = [5, 10, 15]
all_combinations = list(itertools.product(dim_range, walk_len_range, n_walks_range, p_range, q_range, window_range))

# Create a DataFrame to store results
index = pd.MultiIndex.from_tuples(all_combinations, names=['dim', 'walk_len', 'n_walks', 'p', 'q', 'window'])
hopkins_df = pd.DataFrame(index=index, columns=['hopkins'])
mod_df = pd.DataFrame(index=index, columns=['modularity'])
ARI_df = pd.DataFrame(index=index, columns=['ARI'])

for comb in tqdm(all_combinations):
    model = GraphEmbd(G)
    dim, walk_len, n_walks, p, q, window = comb
    model.embd_init(dimensions = dim,
                    walk_length = walk_len,
                    num_walks = n_walks,
                    workers = 1,
                    p = p,
                    q = q,
                    seed = 4,
                    quiet=True)
    model.fit(window=window)
    model.umap(n_jobs=1, reduction='node2vec', n_neighbors=(model.n_nodes-1 if model.n_nodes < 16 else 15))
    
    hp = hopkins(model.reduction['UMAP'], sampling_size=min(15, model.n_nodes))
    hopkins_df.loc[comb, 'hopkins'] = hp
    
    model.kmeans(n_clusters=k_hhc, reduction='UMAP')
    mod_df.loc[comb, 'modularity'] = model.metric['modularity']['KMeans']
    
    ARI_df.loc[comb, 'ARI'] = adjusted_rand_score(model.node_label['KMeans'], prod_nodel_label)
        
hopkins_df = hopkins_df.reset_index()
hopkins_df.to_csv(f'Result/hopkins_{HHC_id}.csv')
mod_df = mod_df.reset_index()
mod_df.to_csv(f'Result/mod_{HHC_id}.csv')
ARI_df = ARI_df.reset_index()
ARI_df.to_csv(f'Result/ARI_{HHC_id}.csv')