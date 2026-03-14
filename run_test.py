import pandas as pd
from src.graph_analysis import SentinelGraph
from src.features import FeatureEngineer
from src.model import FraudDetector

def main():
    print("🚀 --- FULL PIPELINE TEST: SentinelGraph AI ---")
    
    # 1. Inizializzazione di tutti i componenti
    graph_engine = SentinelGraph(n_neighbors=5, threshold=0.5)
    engineer = FeatureEngineer()
    # Impostiamo contamination al 5% per vedere qualche anomalia nel test
    detector = FraudDetector(contamination=0.05)
    
    # 2. Caricamento dati
    print("📊 Caricamento dati...")
    raw_data = pd.read_csv('data/creditcard.csv').head(2000) # Alziamo a 2000 per dare più respiro all'IA
    
    # 3. Feature Engineering (Pulizia + Grafo)
    print("🧼 Pulizia e Arricchimento con Grafi...")
    clean_df = engineer.clean_data(raw_data)
    graph_engine.build_similarity_graph(clean_df)
    graph_features = graph_engine.extract_graph_features()
    
    # Unione finale
    final_df = engineer.merge_graph_features(clean_df, graph_features)
    
    # 4. Addestramento e Predizione
    # Prepariamo le feature per il modello (escludendo l'eventuale colonna Class se presente)
    X = final_df.drop(columns=['Class']) if 'Class' in final_df.columns else final_df
    
    detector.train(X)
    predictions = detector.predict(X) # 1 = Normale, -1 = Anomalia
    scores = detector.get_scores(X)   # Più è basso, più è sospetto
    
    # 5. Risultati
    final_df['prediction'] = predictions
    final_df['anomaly_score'] = scores
    
    anomalies_found = (predictions == -1).sum()
    
    print(f"\n✅ ANALISI COMPLETATA!")
    print(f"🔍 Anomalie rilevate: {anomalies_found} su {len(final_df)} transazioni.")
    
    # Vediamo le prime 5 transazioni sospette
    print("\n--- Top 5 Transazioni Sospette (per Anomaly Score) ---")
    print(final_df[final_df['prediction'] == -1][['pagerank', 'clustering', 'anomaly_score']].head())
# salvatagigo del modello
    detector.save_model('models/sentinel_v1.pkl')

if __name__ == "__main__":
    main()