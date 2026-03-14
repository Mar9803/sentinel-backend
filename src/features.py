
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    """
    Gestisce la pulizia dei dati e l'unione tra feature grezze 
    e feature generate dal grafo.
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def clean_data(self, df: pd.DataFrame):
        """Rimuove colonne inutili e gestisce i valori mancanti."""
        # Per il dataset creditcard, rimuoviamo 'Time' se non serve
        if 'Time' in df.columns:
            df = df.drop(columns=['Time'])
        return df.fillna(0)

    def merge_graph_features(self, original_df, graph_features_df):
        """Unisce il dataset originale con le metriche del grafo (PageRank, etc)."""
        # Reset indici per garantire che l'unione sia corretta
        original_df = original_df.reset_index(drop=True)
        #contateno per colonne, ottenog un dataFrame unico: feat_grezzze+feat_topologiche
        graph_features_df = graph_features_df.reset_index(drop=True)
        
        return pd.concat([original_df, graph_features_df], axis=1)








"""
import pandas as pd
import numpy as np
#PRIMA FEATURE: velocity: quante transazioni in un intervallo di tempo. Scopo: scoprire se stanno avvenendo bruteforce.
def extract_velocity_features(df):
    #Calcola il numero di transazioni nelle ultime 1, 6 e 24 ore per ogni utente.
    df = df.sort_values(['user_id', 'timestamp'])
    # Trasformiamo il timestamp in formato datetime se non lo è
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Impostiamo l'indice temporale per usare i metodi di windowing
    df = df.set_index('timestamp')
    # Conteggio transazioni nelle ultime 1h (finestra mobile)
    df['tx_count_1h'] = df.groupby('user_id')['amount'].rolling('1H').count().values
    # Media dell'importo speso nelle ultime 24h
    df['avg_amount_24h'] = df.groupby('user_id')['amount'].rolling('24H').mean().values
    return df.reset_index()
    
"""
