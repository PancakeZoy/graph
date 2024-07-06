import pickle
import pandas as pd
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '..'))
sys.path.append(parent_dir)
from GraphEmbd.utils import NodeLabel_to_communities

with open('result.pkl', 'rb') as f:
    result = pickle.load(f)

prod_result = pd.read_csv('../Data/ind_CoreHH.csv')
edges = pd.read_csv('../Data/edge_list.csv')
edges.columns = ['source', 'target', 'weight', 'hhcluster_id']
HHC_size = pd.read_csv('../Data/HH_size.csv')
HHC_id_set = list(HHC_size.hhcluster_id)
edges_split = {HHC_id: edges[edges.hhcluster_id==HHC_id] for HHC_id in HHC_id_set}
prod_split = {HHC_id: prod_result[prod_result.hhcluster_id == HHC_id] for HHC_id in HHC_id_set}

km_comm = []; lv_comm = []
km_df = prod_result.set_index('ind_id'); km_df.hhcluster_id = None
lv_df = prod_result.set_index('ind_id'); lv_df.hhcluster_id = None
for hhc_id, value in result.items():
    edges_sub = edges_split[hhc_id]
    km_sub_comm = value['node_label']['KMeans']
    km_sub_comm = NodeLabel_to_communities(list(km_sub_comm.values()), list(km_sub_comm.keys()))
    km_comm.extend(km_sub_comm)
    
    lv_sub_comm = value['node_label']['Louvian']
    lv_sub_comm = NodeLabel_to_communities(list(lv_sub_comm.values()), list(lv_sub_comm.keys()))
    lv_comm.extend(lv_sub_comm)
    
for cluster_id, nodes in enumerate(km_comm):
    km_df.loc[list(nodes), 'hhcluster_id'] = cluster_id
    
for cluster_id, nodes in enumerate(lv_comm):
    lv_df.loc[list(nodes), 'hhcluster_id'] = cluster_id
    

