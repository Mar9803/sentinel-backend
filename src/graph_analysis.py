import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class SentinelGraph:
    def __init__(self, n_neighbors=5, threshold=0.5):
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.G = nx.Graph()

    def build_similarity_graph(self, df):
        # Prendiamo solo colonne numeriche per il grafo
        features = df.select_dtypes(include=['float64', 'int64'])
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(scaled_features)
        distances, indices = nn.kneighbors(scaled_features)
        
        self.G.clear()
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:
                self.G.add_edge(i, neighbor)

    def extract_graph_features(self):
        pagerank = nx.pagerank(self.G)
        clustering = nx.clustering(self.G)
        return pd.DataFrame({
            'pagerank': pagerank.values(),
            'clustering': clustering.values()
        })