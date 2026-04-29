from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import joblib

# Importiamo i tuoi componenti dalla cartella src
from src.graph_analysis import SentinelGraph
from src.features import FeatureEngineer
from src.model import FraudDetector

app = FastAPI(title="SentinelGraph API")

# --- CONFIGURAZIONE CORS ---
# Permette ad Astro (solitamente su porta 4321 o 3000) di comunicare col Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In produzione metti l'URL del tuo blog Astro
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inizializzazione detector, graph_engine, engineer
detector= FraudDetector()

# --- INIZIALIZZAZIONE COMPONENTI ---
# Carichiamo il modello una sola volta all'avvio per risparmiare memoria e tempo
try:
    # Qui usiamo la logica del tuo FraudDetector 
    detector.load_model('models/sentinel_v1.pkl')
    print("✅ Modello caricato tramite metodo di classe")
except Exception as e:
    print(f"❌ Errore: {e}")

# Inizializziamo gli altri motori
graph_engine = SentinelGraph(n_neighbors=5, threshold=0.5)
engineer = FeatureEngineer()

@app.get("/")
def root():
    return {"status": "Il Beckend è vivo", "project": "SentinelGraph AI"}

# ESEMPIO ENDPOINT PER I GRAFICI
@app.get("/api/stats")
async def get_stats():
    """Endpoint veloce per il blog Astro"""
    return {
        "status": "online",
        "engine": "Isolation Forest + NetworkX",
        "alerts_today": 12, # Esempio di dato statico o recuperato da DB
        "risk_levels": {"Safe": 85, "Warning": 10, "Fraud": 5}
    }


@app.post("/analyze")
async def analyze_transactions(file: UploadFile = File(...)):
    # 1. Controllo estensione file prima del try: errore web comunicato nel browser.
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Il file deve essere un CSV")
    
    if not detector.is_trained:
            raise HTTPException(status_code=503, detail="Modello ML non disponibile sul server")
    try:
        # 2. Lettura del file inviato tramite fetch
        contents = await file.read()
        raw_data = pd.read_csv(io.BytesIO(contents))
        
        # 3. PIPELINE, fase dei calcoli ML+NetworkX
        # Pulizia
        clean_df = engineer.clean_data(raw_data)
        # Analisi Grafo
        graph_engine.build_similarity_graph(clean_df)
        graph_features = graph_engine.extract_graph_features()
        # Merge
        final_df = engineer.merge_graph_features(clean_df, graph_features)
        # --- 4. PREDIZIONE (VERSIONE FINALE CORAZZATA) ---
        # 1. Definiamo i gruppi di colonne
        v_cols = [f'V{i}' for i in range(1, 29)] # V1...V28
        graph_cols = ['pagerank', 'clustering', 'betweenness']
        
        # 2. Questo è l'ordine esatto che il modello si aspetta (V1-V28, poi Amount, poi Grafo)
        expected_order = v_cols + ['Amount'] + graph_cols
        
        # 3. Prepariamo il DataFrame X partendo da final_df
        X = final_df.copy()
        
        # 4. Tappiamo i buchi (se mancano colonne nel CSV o nel Grafo)
        for col in expected_order:
            if col not in X.columns:
                X[col] = 0.0
        
        # 5. FORZIAMO L'ORDINE (Questa è la riga che risolve l'errore del terminale)
        X = X[expected_order]
        
        # 6. Predizione
        predictions = detector.predict(X)
        scores = detector.get_scores(X) 
        
        # Riattacchiamo i risultati a final_df per la visualizzazione
        final_df['prediction'] = predictions
        final_df['anomaly_score'] = scores
        
        # --- STATISTICHE ---
        counts, bins = np.histogram(scores, bins=10)
        risk_counts = {
            "Safe": int((scores > 0.05).sum()),
            "Warning": int(((scores <= 0.05) & (scores > -0.05)).sum()),
            "Fraud": int((scores <= -0.05).sum())
        }
    
        sospetti = final_df[final_df['prediction'] == -1].head(10)
     
        return {
            "total_analyzed": len(final_df),
            "anomalies_found": int((predictions == -1).sum()),
            "stats": {
                "risk_distribution": risk_counts,
                "histogram": {
                    "labels": [f"{round(b, 2)}" for b in bins[:-1]],
                    "values": counts.tolist()
                }
            },
            "results": sospetti[['pagerank', 'clustering', 'anomaly_score']].to_dict(orient='records')
        }

    except Exception as e:
        # Stampiamo l'errore nel terminale per debuggare meglio
        print(f"ERRORE DURANTE ANALISI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)