import pandas as pd
import random
import time
import multiprocessing
from worker import embd_clust

def main():
    prod_result = pd.read_csv('../Data/ind_CoreHH.csv')
    edges = pd.read_csv('../Data/edge_list.csv')
    edges.columns = ['source', 'target', 'weight', 'hhcluster_id']
    HHC_size = pd.read_csv('../Data/HH_size.csv')
    random.seed(35)
    HHC_id_set = random.sample(list(HHC_size.hhcluster_id), 100)

    edges_split = {HHC_id: edges[edges.hhcluster_id==HHC_id] for HHC_id in HHC_id_set}
    prod_split = {HHC_id: prod_result[prod_result.hhcluster_id == HHC_id] for HHC_id in HHC_id_set}
    args = list(zip(HHC_id_set, 
                    [edges_split[i] for i in HHC_id_set],
                    [prod_split[i] for i in HHC_id_set]))

    with multiprocessing.Pool(processes=5) as pool:
        results = pool.starmap(embd_clust, args)
    result = dict(zip(HHC_id_set, results))
    return result

start = time.time()
if __name__ == "__main__":
    result = main()
end = time.time()
print(end - start)