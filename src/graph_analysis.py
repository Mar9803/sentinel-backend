import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

@app.get("/")
def root():
    return {"status": "online", "project": "SentinelGraph AI"}

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
        # 4. PREDIZIONE (usando il modello caricato)
        X = final_df.drop(columns=['Class']) if 'Class' in final_df.columns else final_df
        # Usiamo il modello già pronto
        predictions = detector.predict(X)
        scores = detector.get_scores(X) # o uso metodo get_scores(X) nella classe 
        
        # 5. PREPARAZIONE RISPOSTA PER ASTRO
        # Creiamo un risultato leggibile dal frontend
        final_df['prediction'] = predictions
        final_df['anomaly_score'] = scores
        
        # Filtriamo solo le anomalie per non mandare troppi dati, o mandiamo tutto
        # Per ora mandiamo i top 10 sospetti come esempio
        sospetti = final_df[final_df['prediction'] == -1].head(10)
        return {
            "total_analyzed": len(final_df),
            "anomalies_found": int((predictions == -1).sum()),
            "results": sospetti[['pagerank', 'clustering', 'anomaly_score']].to_dict(orient='records')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
