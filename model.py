import numpy as np
import networkx as nx
from node2vec import Node2Vec
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from utils import label_to_communities
import matplotlib.pyplot as plt

class GraphEmbd():
    def __init__(
            self,
            graph):
        self.G = graph
        self.nodes = np.array(graph.nodes)
        self.n_nodes = len(self.nodes)
        self.node_label = {}
        self.node2vec_model = None
        self.clust_model = None
        self.reduction = {}
        self.metric = {}
        self.trace = {}
        self.n_trigs = sum(nx.triangles(graph).values())
        self.grid = {}
    
    def embd_init(self, 
                   dimensions = 128,
                   walk_length = 80,
                   num_walks = 10,
                   workers = 1,
                   p = 0.8,
                   q = 1,
                   seed = 42,
                   **kwargs):
        """
        Initialize the Node2Vec model.
        
        Parameters
        ----------
        - graph: 
            Input graph.
        - dimensions: 
            Embedding dimensions (default: 128).
        - walk_length: 
            Number of nodes in each walk (default: 80).
        - num_walks: 
            Number of walks per node (default: 10).
        - workers: 
            Number of workers for parallel execution (default: 1).
        - p: 
            Return parameter (default: 0.8).
        - q: 
            In-out parameter (default: 1).
        - seed: 
            Deterministic results can be obtained if seed is set and workers=1.

        Attributes
        ----------
        - self.node2vec_model: Node2Vec
            The Node2Vec model instance created with the specified parameters. This model can be used 
            to generate node embeddings based on random walks of the input graph.
        """
        self.node2vec_model = Node2Vec(
            graph=self.G,
            dimensions = dimensions,
            walk_length = walk_length,
            num_walks = num_walks,
            workers = workers,
            p = p,
            q = q,
            seed = seed,
            **kwargs
        )
        
    def fit(self,
            window=10,
            min_count=1,
            batch_words=4,
            reduction_name = 'node2vec',
            **kwargs):
        """
        Fit the Node2Vec model by running Word2Vec algorithm.
        
        Parameters
        ----------
        - window: 
            Maximum distance between the current and predicted word within a sentence.
        - min_count: 
            Ignores all words with total frequency lower than this.
        - batch_words: 
            Target size (in words) for batches of examples passed to worker threads 
            (and thus cython routines). Larger batches will be passed if individual texts 
            are longer than 10,000 words, but the standard cython code truncates to that maximum.
        - reduction_name:
            Name of the reduction key used to store embeddings (default: 'node2vec').
            
        Attributes
        ----------
        - self.reduction[reduction_name]: ndarray of shape (n_nodes, n_dimensions)
            The node embedding matrix, row re-ordered to match the node order of self.G 
        """
        self.node2vec_model = self.node2vec_model.fit(
            window=window,
            min_count=min_count,
            batch_words=batch_words,
            **kwargs
            )
        node_embeddings = self.node2vec_model.wv.get_normed_vectors()
        embd_nodes_order = np.array(self.node2vec_model.wv.index_to_key).astype(int)
        order_index = {node: i for i, node in enumerate(embd_nodes_order)}
        node_embeddings = np.array([node_embeddings[order_index[node]] for node in self.nodes])
        self.reduction[reduction_name] = node_embeddings
        
    def pca(self,
            reduction = 'node2vec',
            reduction_name = 'PCA',
            n_components = 10,
            random_state = 42,
            **kwargs):
        """
        Perform PCA on the node embeddings for dimensionality reduction.
        
        Parameters
        ----------
        - reduction: str, optional, default='node2vec'
            The key in the reduction dictionary from which to get the embeddings for PCA.
        - reduction_name: str, optional, default='PCA'
            The key in the reduction dictionary under which to store the PCA embeddings.
        - n_components: int, optional, default=10
            The number of components to keep.
        - random_state: int, optional, default=42
            The seed for random number generation.
        - kwargs: dict, optional
            Additional keyword arguments for PCA.
        
        Attributes
        ----------
        - self.reduction[reduction_name]: ndarray of shape (n_nodes, n_components)
            The PCA embeddings
        """
        reducer = PCA(n_components=n_components, random_state=random_state, **kwargs)
        pcs = reducer.fit_transform(self.reduction[reduction])
        self.reduction[reduction_name] = pcs
        
    
    def umap(self, 
             reduction = 'node2vec',
             reduction_name = 'UMAP',
             n_components = 2,
             metric = 'cosine',
             random_state = 3,
             **kwargs):
        """
        Perform UMAP on the node embeddings for visualization and clustering.
        
        Parameters
        ----------
        - reduction: str, optional, default='node2vec'
            The key in the reduction dictionary from which to get the embeddings for UMAP.
        - reduction_name: str, optional, default='UMAP'
            The key in the reduction dictionary under which to store the UMAP embeddings.
        - n_components: int, optional, default=2
            The number of dimensions to reduce the data to. Typically, for visualization purposes, 
            this is set to 2.
        - metric: str or callable, optional, default='cosine'
            The distance metric to use for computing distances between pairs of samples. 
            It can be any metric supported by `scipy.spatial.distance.pdist` or a user-defined function.
        - random_state: int, RandomState instance or None, optional, default=3
            The seed of the pseudo-random number generator to use for initializing the optimization.
            If an integer is given, it fixes the seed. If None, the random number generator is the 
            RandomState instance used by `np.random`.
        - kwargs: dict, optional
            Additional keyword arguments to pass to the UMAP constructor.
        
        Attributes
        ----------
        - self.reduction[reduction_name]: ndarray of shape (n_nodes, n_components)
            The UMAP embeddings
        """
        
        reducer = UMAP(n_components=n_components, metric=metric, random_state=random_state, **kwargs)
        valid_keys = list(self.reduction.keys())
        if reduction_name in valid_keys:
            valid_keys.remove(reduction_name)
        if reduction not in valid_keys:
            raise ValueError(f"Must give a reduction name among {valid_keys}")
        umap_coor = reducer.fit_transform(self.reduction[reduction])
        self.reduction[reduction_name] = umap_coor
        
    def kmeans(self,
               reduction='PCA',
               n_clusters=4, 
               random_state=42, 
               n_init=15,
               **kwargs):
        """
        Perform KMeans clustering on the reduced coordinates.
    
        Parameters
        ----------
        - reduction: str, optional, default='PCA'
            The key in the reduction dictionary from which to get the embeddings for KMeans clustering.
        - n_clusters: int, optional, default=4
            The number of clusters to form.
        - random_state: int, optional, default=42
            Determines random number generation for centroid initialization.
        - n_init: int, optional, default=15
            Number of times the k-means algorithm will be run with different centroid seeds.
        - kwargs: dict, optional
            Additional keyword arguments for the KMeans constructor.
    
        Attributes
        ----------
        - self.node_label['KMeans']: ndarray of shape (n_nodes,)
            Cluster labels for each point in the dataset.
        """
        model = KMeans(n_clusters=n_clusters, 
                        random_state=random_state, 
                        n_init=n_init, 
                        **kwargs)
        if reduction not in self.reduction.keys():
            raise ValueError(f"Must give a reduction name among {list(self.reduction.keys())}")
        pred_label = model.fit_predict(self.reduction[reduction])
        self.clust_model = model
        self.node_label['KMeans'] = pred_label
        self.modularity('KMeans')

    def louvian(self, seed=123, **kwargs):
        """
        Perform Louvain community detection on the graph.
    
        Parameters
        ----------
        - seed: int, optional, default=123
            Random seed for reproducibility.
        - kwargs: dict, optional
            Additional keyword arguments for the Louvain method.
    
        Attributes
        ----------
        - self.node_label['Louvian']: ndarray of shape (n_nodes,)
            Community labels for each node in the graph.
        """
        louv_comm = nx.community.louvain_communities(self.G, seed=seed)
        nodes_list = list(self.G.nodes)
        node_to_cluster = {}
        for cluster_id, cluster in enumerate(louv_comm):
            for node in cluster:
                node_to_cluster[node] = cluster_id
        louv_label = np.array([node_to_cluster[node] for node in nodes_list]).astype(int)
        self.node_label['Louvian'] = louv_label
        self.modularity('Louvian')
        
    def HDBSCAN(self,
                reduction='PCA',
                min_cluster_size=3,
                max_cluster_size=10,
                **kwargs):
        if reduction not in self.reduction.keys():
            raise ValueError(f"Must give a reduction name among {list(self.reduction.keys())}")
        hdb = HDBSCAN(min_cluster_size=min_cluster_size,
                      max_cluster_size=max_cluster_size)
        hdb.fit(self.reduction[reduction])
        self.node_label['HDBSCAN'] = hdb.labels_
        self.modularity('HDBSCAN')
        
    def hyper_tune(self, grid, method = 'KMeans', reduction = 'PCA'):
        """
        Perform hyperparameter tuning to select the optimal clustering parameters based on various metrics.
        
        Parameters
        ----------
        - grid: array-like
            A list or array specifying the range of k values (for KMeans) or min_cluster_size values (for HDBSCAN) to explore.
        - method: str, optional, default='KMeans'
            The clustering method to use. Options are 'KMeans' and 'HDBSCAN'.
        - reduction: str, optional, default='PCA'
            The key in the reduction dictionary from which to get the embeddings for clustering.
        
        Attributes
        ----------
        - self.trace: dict
            A dictionary containing the following keys for each method:
            - 'modularity': ndarray
                Modularity scores for each value in the grid.
            - 'silhouette': ndarray
                Silhouette scores for each value in the grid.
        
        Notes
        -----
        This method performs clustering using either KMeans or HDBSCAN over a range of parameters specified in the grid.
        For each parameter value, it calculates the modularity and silhouette scores and stores these metrics in the `trace` attribute.
        """
        mod = np.array([]); sil = np.array([]);
        self.grid[method] = grid
        for p in grid:
            if method == 'KMeans':            
                self.kmeans(reduction=reduction, n_clusters=p)
            elif method == 'HDBSCAN':
                self.HDBSCAN(reduction=reduction, min_cluster_size=p)
            else:
                raise ValueError("Invalid method. Please choose either 'KMeans' or 'HDBSCAN'.")
            pred_label = self.node_label[method]
            mod = np.append(mod, nx.community.modularity(self.G, label_to_communities(pred_label, self.nodes)))
            sil = np.append(sil, silhouette_score(self.reduction[reduction], pred_label))
        trace_method = {'modularity': mod, 'silhouette': sil}
        self.trace[method] = trace_method
    
    def modularity(self, method):
        """
        Calculate the modularity of the communities detected by the given method.
    
        Parameters
        ----------
        - method: str
            The name of the method used to detect communities.
    
        Raises
        ------
        - ValueError: If the method is not found in self.node_label.
    
        Attributes
        ----------
        - self.metric['modularity'][method]: float
            The modularity score for the given method.
        """
        if method not in self.node_label.keys():
            raise ValueError(f"Must give a method name among {list(self.node_label.keys())}")
        pred_comm = label_to_communities(self.node_label[method], self.nodes)
        mod_value = nx.community.modularity(self.G, pred_comm)
        if 'modularity' not in self.metric:
            self.metric['modularity'] = {}
        self.metric['modularity'][method] = mod_value
    
    
    def draw_graph(self, method=None, seed=86, title=None, ax=None, **kwargs):
        """
        Draw the graph with nodes colored by the given method.
        
        Parameters
        ----------
        - method: str, optional
            Method name for coloring the nodes.
        - seed: int, optional, default=86
            Random seed for the layout algorithm.
        - title: str, optional
            Title for the graph.
        - ax: matplotlib.axes.Axes, optional
            Matplotlib axes object to plot on. If None, create a new plot.
        - kwargs: dict, optional
            Additional keyword arguments for networkx.draw function.
        """
        pos = nx.spring_layout(self.G, seed=seed)
        
        if method is not None:
            if method not in self.node_label.keys():
                raise ValueError(f"Must give a method name among {list(self.node_label.keys())}")
            node_color = self.node_label[method]
        else:
            node_color = '#1f78b4'  # Default color if no method is provided

        if ax is None:
            fig, ax = plt.subplots()
            show_plot = True
        else:
            show_plot = False

        nx.draw(self.G, pos, ax=ax, node_color=node_color, **kwargs)

        # Set title if provided
        if title is None and method is not None:
            title = f"Graph with {method} labels (Modularity {round(self.metric['modularity'][method], 2)})"
        if title:
            ax.set_title(title)

        if show_plot:
            plt.show()

            
    def plot_embd(self, reduction, with_labels=True, method=None, ax=None, **kwargs):
        """
        Plot reduced embeddings with flexibility for customization.

        Parameters
        ----------
        - reduction: str
            The key in the reduction dictionary from which to get the embeddings for plotting.
        - method: str, optional
            Method name for coloring the points.
        - ax: matplotlib.axes.Axes, optional
            Matplotlib axes object to plot on. If None, create a new plot.
        - kwargs: dict, optional
            Additional keyword arguments for customization (e.g., axes labels, savefig).
        """
        if method is not None:
            if method not in self.node_label.keys():
                raise ValueError(f"Must give a method name among {list(self.node_label.keys())}")
            colors = self.node_label[method]
        else:
            colors = None

        if reduction not in self.reduction.keys():
            raise ValueError(f"Must give a reduction name among {list(self.reduction.keys())}")

        # Extract specific keys from kwargs
        xlabel = kwargs.pop('xlabel', f'{reduction} 1')
        ylabel = kwargs.pop('ylabel', f'{reduction} 2')
        title = kwargs.pop('title', None)
        savefig = kwargs.pop('savefig', None)
            
        # Create the plot
        if ax is None:
            fig, ax = plt.subplots()
            show_plot = True
        else:
            show_plot = False
    
        # Scatter plot
        ax.scatter(self.reduction[reduction][:, 0], self.reduction[reduction][:, 1], c=colors, **kwargs)
    
        # Annotate nodes
        if with_labels:
            for i in range(len(self.nodes)):
                ax.text(self.reduction[reduction][i, 0], self.reduction[reduction][i, 1], s=str(self.nodes[i]), 
                        horizontalalignment='center', verticalalignment='center')
    
        # Customize axes if provided
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
    
        # Save the figure if 'savefig' is provided
        if savefig and ax is None:
            plt.savefig(savefig)
    
        # Show the plot if no axes object is provided
        if show_plot:
            plt.show()


    def plot_trace(self, method, score, ax=None, **kwargs):
        """
        Plot a specified score array in self.trace as a line scatter plot.
    
        Parameters
        ----------
        - score: str
            The key of the score array in self.trace to plot.
        - ax: matplotlib.axes.Axes, optional
            Matplotlib axes object to plot on. If None, create a new plot.
        - kwargs: dict, optional
            Additional keyword arguments for customization (e.g., xlabel, ylabel, title, savefig).
    
        Raises
        ------
        - ValueError: If the specified score is not found.
        """
        if method not in self.trace:
            raise ValueError(f"The specified method '{method}' is not found in self.trace. Available methods are: {list(self.trace.keys())}")            
        if score not in self.trace[method]:
            raise ValueError(f"The specified score '{score}' is not found in self.trace[{method}] Available scores are: {list(self.trace[method].keys())}")
    
        # Extract specific keys from kwargs
        xlabel = kwargs.pop('xlabel', 'Grid')
        ylabel = kwargs.pop('ylabel', score.capitalize())
        title = kwargs.pop('title', f'{score.capitalize()} Scores for Different K Values')
        savefig = kwargs.pop('savefig', None)
    
        # Create the plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            show_plot = True
        else:
            show_plot = False
    
        # Plot the specified score array
        values = self.trace[method][score]
        ax.plot(self.grid[method], values, marker='o', label=score, **kwargs)
    
        # Customize the plot
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    
        # Save the figure if 'savefig' is provided
        if savefig and ax is None:
            plt.savefig(savefig)
    
        # Show the plot if no axes object is provided
        if show_plot:
            plt.show()
