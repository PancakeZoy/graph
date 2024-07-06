import pandas as pd
import random
import time
import multiprocessing
import pickle
from worker import embd_clust

def main():
    prod_result = pd.read_csv('../Data/ind_CoreHH.csv')
    edges = pd.read_csv('../Data/edge_list.csv')
    edges.columns = ['source', 'target', 'weight', 'hhcluster_id']
    HHC_size = pd.read_csv('../Data/HH_size.csv')
    random.seed(35)
    HHC_id_set = random.sample(list(HHC_size.hhcluster_id), 100)
    # HHC_id_set = list(HHC_size.hhcluster_id)

    edges_split = {HHC_id: edges[edges.hhcluster_id==HHC_id] for HHC_id in HHC_id_set}
    prod_split = {HHC_id: prod_result[prod_result.hhcluster_id == HHC_id] for HHC_id in HHC_id_set}
    args = list(zip(HHC_id_set, 
                    [edges_split[i] for i in HHC_id_set],
                    [prod_split[i] for i in HHC_id_set]))

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(embd_clust, args)
    result = dict(zip(HHC_id_set, results))
    return result

start = time.time()
if __name__ == "__main__":
    result = main()
end = time.time()
print(end - start)

####################################################################################
####################################################################################
####################################################################################
# start = time.time()
# prod_result = pd.read_csv('../Data/ind_CoreHH.csv')
# edges = pd.read_csv('../Data/edge_list.csv')
# edges.columns = ['source', 'target', 'weight', 'hhcluster_id']
# HHC_size = pd.read_csv('../Data/HH_size.csv')
# random.seed(35)
# HHC_id_set = random.sample(list(HHC_size.hhcluster_id), 100)

# edges_split = {HHC_id: edges[edges.hhcluster_id==HHC_id] for HHC_id in HHC_id_set}
# prod_split = {HHC_id: prod_result[prod_result.hhcluster_id == HHC_id] for HHC_id in HHC_id_set}
# args = list(zip(HHC_id_set, 
#                 [edges_split[i] for i in HHC_id_set],
#                 [prod_split[i] for i in HHC_id_set]))
# result = {}
# for arg in args:
#     HHC_id, edges_sub, prod_sub = arg
#     result[HHC_id] = embd_clust(HHC_id, edges_sub, prod_sub)
# end = time.time()
# print(end - start)

# with open('result.pkl', 'wb') as f:
#     pickle.dump(result, f)