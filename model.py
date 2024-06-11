import numpy as np
import networkx as nx
from node2vec import Node2Vec
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import label_to_communities
import matplotlib.pyplot as plt

class GraphEmbd():
    def __init__(
            self,
            graph):
        self.G = graph
        self.nodes = np.array(graph.nodes)
        self.n_nodes = len(self.nodes)
        self.node_attr = {'True': np.fromiter(nx.get_node_attributes(graph, 'label').values(), dtype=int)}
        self.node2vec_model = None
        self.clust_model = None
        self.embedding = None
        self.umap_coor = None
        self.metric = {}
        self.trace_K = {}
        self.modularity('True')
    
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
            Return parameter (default: 1).
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
            
        Attributes
        ----------
        - self.embedding: ndarray of shape (n_nodes, n_dimensions)
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
        self.embedding = node_embeddings
        
    def umap(self, 
             n_components = 2,
             metric = 'cosine',
             random_state = 3,
             **kwargs):
        """
        Perform UMAP on the node embeddings for visualization and clustering.
        
        Parameters
        ----------
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
        
        - **kwargs: dict, optional
            Additional keyword arguments to pass to the UMAP constructor. These can include parameters
            such as `n_neighbors`, `min_dist`, `spread`, `set_op_mix_ratio`, `local_connectivity`, etc.
        
        Attributes
        ----------
        - self.umap_coor: ndarray of shape (n_nodes, n_components)
            The transformed data after applying UMAP.
        """
    
        reducer = UMAP(n_components=n_components, metric=metric, random_state=random_state, **kwargs)
        umap_coor = reducer.fit_transform(self.embedding)
        self.umap_coor = umap_coor
        
    def kmeans(self,
               n_clusters=4, 
               random_state=42, 
               n_init=15,
               **kwargs):
        """
        Perform KMeans clustering on the UMAP coordinates.
    
        Parameters
        ----------
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
        - self.node_attr['KMeans']: ndarray of shape (n_nodes,)
            Cluster labels for each point in the dataset.
        """
        model = KMeans(n_clusters=n_clusters, 
                        random_state=random_state, 
                        n_init=n_init, 
                        **kwargs)
        pred_label = model.fit_predict(self.umap_coor)
        self.clust_model = model
        self.node_attr['KMeans'] = pred_label
        self.modularity('KMeans')
        
    def k_selection(self, k_min_max=None):
        """
        Perform selection of the optimal number of clusters (k) based on various metrics.
    
        Parameters
        ----------
        - k_min_max: tuple of int, optional
            A tuple (k_min, k_max) specifying the range of k values to explore.
            If None, defaults to a range between max(2, n_nodes/10) and min(n_nodes/2, n_nodes).
    
        Attributes
        ----------
        - self.trace_K: dict
            A dictionary containing the following keys:
            - 'modularity': ndarray
                Modularity scores for each k value.
            - 'wss': ndarray
                Within-cluster sum of squares (inertia) for each k value.
            - 'silhouette': ndarray
                Silhouette scores for each k value.
    
        Notes
        -----
        This method uses KMeans clustering to compute the cluster labels for a range of k values. 
        For each k, it calculates the modularity, within-cluster sum of squares (WSS), and silhouette score, 
        and stores these metrics in the `trace_K` attribute.
        """        
        if k_min_max is None:            
            k_min = max(2, int(self.n_nodes/10))
            k_max = min(int(self.n_nodes/2), self.n_nodes)
        else:
            k_min = k_min_max[0]
            k_max = k_min_max[1]
        k_grid = np.arange(k_min, k_max)
        self.k_grid = k_grid
        mod = np.array([]); wss = np.array([]); sil = np.array([]);
        for k in k_grid:
            self.kmeans(n_clusters=k)
            pred_label = self.node_attr['KMeans']
            mod = np.append(mod, nx.community.modularity(self.G, label_to_communities(pred_label, self.nodes)))
            wss = np.append(wss, self.clust_model.inertia_)
            sil = np.append(sil, silhouette_score(self.umap_coor, pred_label))
        self.trace_K = {'modularity': mod,
                        'wss': wss,
                        'silhouette': sil}
            
    
    def modularity(self, method):
        """
        Calculate the modularity of the communities detected by the given method.
    
        Parameters
        ----------
        - method: str
            The name of the method used to detect communities.
    
        Raises
        ------
        - ValueError: If the method is not found in self.node_attr.
    
        Attributes
        ----------
        - self.metric['modularity'][method]: float
            The modularity score for the given method.
        """
        if method not in self.node_attr.keys():
            raise ValueError(f"Must give a method name among {list(self.node_attr.keys())}")
        pred_comm = label_to_communities(self.node_attr[method], self.nodes)
        mod_value = nx.community.modularity(self.G, pred_comm)
        if 'modularity' not in self.metric:
            self.metric['modularity'] = {}
        self.metric['modularity'][method] = mod_value
    
    
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
        - self.node_attr['Louvian']: ndarray of shape (n_nodes,)
            Community labels for each node in the graph.
        """        
        louv_comm = nx.community.louvain_communities(self.G, seed=seed)
        louv_label = np.ones(len(self.G.nodes))
        for label, nodes in enumerate(louv_comm):
            louv_label[list(nodes)] = label
        self.node_attr['Louvian'] = louv_label[self.nodes].astype(int)
        self.modularity('Louvian')
    
    
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
            if method not in self.node_attr.keys():
                raise ValueError(f"Must give a method name among {list(self.node_attr.keys())}")
            node_color = self.node_attr[method]
        else:
            node_color = 'blue'  # Default color if no method is provided

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

            
    def plot_embd_umap(self, method=None, ax=None, **kwargs):
        """
        Plot UMAP embedding with flexibility for customization.

        Parameters
        ----------
        - method: 
            Method name for coloring the points.
        - ax: 
            Matplotlib axes object to plot on. If None, create a new plot.
        - kwargs:
            Additional keyword arguments for customization (e.g., axes labels, savefig).
        """
        if method is not None:
            if method not in self.node_attr.keys():
                raise ValueError(f"Must give a method name among {list(self.node_attr.keys())}")
            colors = self.node_attr[method]
        else:
            colors = None

        # Extract specific keys from kwargs
        xlabel = kwargs.pop('xlabel', 'UMAP 1')
        ylabel = kwargs.pop('ylabel', 'UMAP 2')
        title = kwargs.pop('title', None)
        savefig = kwargs.pop('savefig', None)
            
        # Create the plot
        if ax is None:
            fig, ax = plt.subplots()
            show_plot = True
        else:
            show_plot = False
    
        # Scatter plot
        ax.scatter(self.umap_coor[:, 0], self.umap_coor[:, 1], c=colors, **kwargs)
    
        # Annotate nodes
        for i in range(len(self.nodes)):
            ax.text(self.umap_coor[i, 0], self.umap_coor[i, 1], s=str(self.nodes[i]), 
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


    def plot_trace_K(self, score, ax=None, **kwargs):
        """
        Plot a specified score array in self.trace_K as a line scatter plot.
    
        Parameters
        ----------
        - score: str
            The key of the score array in self.trace_K to plot.
        - ax: matplotlib.axes.Axes, optional
            Matplotlib axes object to plot on. If None, create a new plot.
        - kwargs: dict, optional
            Additional keyword arguments for customization (e.g., xlabel, ylabel, title, savefig).
    
        Raises
        ------
        - ValueError: If self.trace_K is not defined or the specified score is not found.
        """
        if not hasattr(self, 'trace_K'):
            raise ValueError("self.trace_K is not defined.")
        if score not in self.trace_K:
            raise ValueError(f"The specified score '{score}' is not found in self.trace_K. Available scores are: {list(self.trace_K.keys())}")
    
        # Extract specific keys from kwargs
        xlabel = kwargs.pop('xlabel', 'Number of clusters K')
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
        values = self.trace_K[score]
        ax.plot(self.k_grid, values, marker='o', label=score, **kwargs)
    
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