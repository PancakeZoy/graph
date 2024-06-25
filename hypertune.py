from model import GraphEmbd
import networkx as nx
from utils import *
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import random
from tqdm import tqdm

prod_result = pd.read_csv('SampleGraph/xref_ind_to_core_hh_intern_snapshot_20240314_sample01.csv', skiprows=4)
prod_result = prod_result.iloc[:-2,]
prod_result['ind_id'] = prod_result['ind_id'].astype('int64')
prod_result['hhcluster_id'] = prod_result['hhcluster_id'].astype('int64')
prod_result['core_hh_id'] = prod_result['core_hh_id'].astype('int64')
prod_result.head()

edges = pd.read_csv('SampleGraph/ind_graph_edge_list_snapshot_20240314_sample01.csv', skiprows=3)
edges.columns = ['source', 'target', 'weight', 'HouseHold_id']
edges = edges.iloc[:-2,]
edges['source'] = edges['source'].astype('int64')
edges['target'] = edges['target'].astype('int64')
edges['HouseHold_id'] = edges['HouseHold_id'].astype('int64')
edges.head()

nodes_in_prod=prod_result.ind_id.unique()
nodes_in_edge = np.unique(list(edges.source.unique()) + list(edges.target.unique()))
print(f'{len(nodes_in_prod)} unique nodes found in the production result file')
print(f'{len(nodes_in_edge)} unique nodes found in the edge list')
print(f'{len(np.intersect1d(nodes_in_prod, nodes_in_edge))} matched nodes found')

HHC_size = pd.DataFrame(edges.HouseHold_id.value_counts())
random.seed(42)
HH_id_set = random.sample(list(HHC_size[HHC_size.HouseHold_id>50].index), 50)

for hh_index, hh_id in enumerate(HH_id_set):
    edges_HH = edges[edges.HouseHold_id==hh_id]
    G = nx.from_pandas_edgelist(edges_HH, 'source', 'target', edge_attr='weight')
    concom_all = sorted(list(nx.connected_components(G)), key=len, reverse=True)
    concom_size = np.array([len(cc) for cc in concom_all])
    concom_set = [concom_all[i] for i in np.where(concom_size>=15)[0]]
    
    for cc_index, concom in enumerate(concom_set):
        sub_G = G.subgraph(concom)
        model = GraphEmbd(sub_G)

        # dim_range = np.arange(5, 105, 5)
        # walk_len_range = np.arange(5, 85, 5)
        # n_walks_range = np.arange(5, 15)
        # p_range = np.arange(0.5, 1.1, 0.1)
        # q_range = np.arange(0.5, 1.1, 0.1)
        # window_range = np.arange(2, 11, 1)
        dim_range = range(10, 110, 10)
        walk_len_range = range(5, 85, 10)
        n_walks_range = [5, 10, 15]
        p_range = [0.5,0.75, 1.0]
        q_range = [0.5,0.75, 1.0]
        window_range = [5, 10, 15]
        all_combinations = list(itertools.product(dim_range, 
                                                  walk_len_range, 
                                                  n_walks_range, 
                                                  p_range, 
                                                  q_range, 
                                                  window_range))
        # Create a DataFrame to store results
        index = pd.MultiIndex.from_tuples(all_combinations, names=['dim', 'walk_len', 'n_walks', 'p', 'q', 'window'])
        results_df = pd.DataFrame(index=index, columns=['result'])
        for comb in tqdm(all_combinations, desc=f'HH: {hh_index}/50 | ConnComp: {cc_index}/{len(concom_set)}'):
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
            model.umap(n_jobs=1, reduction='node2vec')
            # model.plot_embd(title=all_combinations[i], with_labels=False)
            hp = hopkins(model.reduction['UMAP'], sampling_size=min(15, model.n_nodes))
            results_df.loc[comb, 'result'] = hp
            results_df_reset = results_df.reset_index()
        results_df_reset.to_csv(f'tune/{hh_id}_{cc_index}.csv')
# results_df_reset = results_df.reset_index()
# plt.figure(figsize=(10, 6))
# sns.barplot(data=results_df_reset, x='q', y='result')
# plt.title('Average Result by Dimension (dim)')
# plt.ylabel('Average Result')
# plt.xlabel('Dimension (dim)')
# plt.xticks(rotation=45)
# plt.show()
