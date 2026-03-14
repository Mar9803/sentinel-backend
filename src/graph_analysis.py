import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class SentinelGraph:
    """
    Motore matematico per la trasformazione di log in grafi di simiglianza
    e l'estrazione di feature topologiche per l'Anomaly Detection.
    """
    
    def __init__(self, n_neighbors=5, threshold=0.5):
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.graph = nx.Graph()
        self.scaler = StandardScaler() #per scalare le features prima del KNN.
        self.features_df = None

    def build_similarity_graph(self, data: pd.DataFrame):
        """
        Trasforma un DataFrame in un Grafo KNN filtrato.
        Fase 1 della Roadmap: Il Motore Matematico.
        """
        # 1. Preprocessing e Scaling
        # Escludiamo Time e Class se presenti per il calcolo delle distanze
        cols_to_drop = [c for c in ['Time', 'Class'] if c in data.columns]
        X = data.drop(columns=cols_to_drop)
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Calcolo KNN per popolare distances e indices--> construzione grafo
        knn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        knn.fit(X_scaled)
        distances, indices = knn.kneighbors(X_scaled)
        
        # 3. Costruzione Grafo con Filtraggio
        self.graph.clear()
        for i in range(len(indices)): #nodi
            for j in range(1, self.n_neighbors + 1): #vicini alnodo corrente i
                neighbor_idx = indices[i][j]
                distance = distances[i][j]
                weight = 1 / (1 + distance)
                
                # Applicazione del filtraggio discusso nel Notebook
                if weight >= self.threshold:
                    self.graph.add_edge(i, neighbor_idx, weight=weight)
        
        # Assicuriamoci che tutti i nodi siano presenti (anche se isolati)
        self.graph.add_nodes_from(range(len(data)))
        return self.graph

    def extract_graph_features(self):
        """
        Calcola PageRank, Clustering e Betweenness.
        Queste diventeranno le feature per la Isolation Forest.
        """
        print("Calcolo metriche del grafo in corso...")
        
        pagerank = nx.pagerank(self.graph, weight='weight')
        clustering = nx.clustering(self.graph, weight='weight')
        # Betweenness approssimata per efficienza industriale
        betweenness = nx.betweenness_centrality(self.graph, k=50, weight='weight')
        
        self.features_df = pd.DataFrame({
            'pagerank': pd.Series(pagerank),
            'clustering': pd.Series(clustering),
            'betweenness': pd.Series(betweenness)
        }).fillna(0)
        
        return self.features_df