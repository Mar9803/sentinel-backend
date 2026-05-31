# Sentinel: Real-Time Fraud Detection Ecosystem

## Documento 1: Architettura Globale e Flussi di Rete

Questo documento rappresenta la "Bibbia" architetturale dell'ecosistema Sentinel. Serve a guidare l'Agente nella comprensione delle macro-componenti e dei messaggi ingegneristici del progetto.

---

## 1. I 5 Livelli dell'Ecosistema (Dallo Schema Tecnico)

Il sistema è strutturato secondo un'architettura MLOps standard suddivisa in 5 macro-layer:

### A. Serving Layer (Online)

È la catena che elabora la richiesta in tempo reale.

- **Frontend**: Interfaccia utente (Astro + FastHTML/Streamlit) che invia i dati transazionali.

- **Backend API**: Servizio asincrono FastAPI che riceve la richiesta.

- **Feature Store Online**: Recupera il vettore di feature in tempo reale.

- **Fraud Engine (Modelli Attivi)**: Il modulo che esegue le predizioni in parallelo.

- **Decision Logic**: Applica le regole di business e decide l'esito (Pass / Alert / Block).

### B. Feature Layer

- **Feature Store Online**: Ottimizzato per bassa latenza.

- **Feature Store Offline (PostgreSQL)**: Contiene lo storico completo dei dati, utilizzato per il tracciamento dei dataset storici, training e validazione.

### C. Model Layer (MLflow Tracking)

- **Model Store**: Traccia versioni dei modelli, metriche, parametri e tag.

- **Model Comparison**: Mette a confronto i modelli attivi: *Rules Engine*, *XGBoost vX*, *Autoencoder XY*. Gestisce i rollback in produzione se degradano.

### D. Training Layer (Batch)

- Pipeline ciclica che estrae i dati offline, esegue il Feature Engineering, avvia il training multiplo, valida i modelli, li registra su MLflow e fa la transizione di stadio per il deploy in produzione.

### E. Monitoring Layer

- Controlla la salute del sistema monitorando: *Data Drift*, *Model Drift*, *Performance*, *Explainability (SHAP)* e attiva l'*Alerting* in caso di anomalie di sistema.

---

## 2. Filosofia di Sviluppo e Messaggi Chiave per l'Utente

L'ecosistema deve dimostrare pragmatismo ingegneristico tramite tre messaggi visivi nella Dashboard:

1. **Il Mito dell'ML (Regole vs Modelli):** Non serve sempre il Machine Learning. Le regole statiche (es. importi folli, nazioni impossibili) scremano i pattern evidenti a costo zero computazionale.

2. **Il Paradosso dell'Accuratezza:** Nell'Anomaly Detection, l'accuratezza pura non conta nulla. Se si ha una sola frode da 1 milione di euro su 10.000 transazioni, un modello accurato al 99.9% che la manca è fallimentare. Focus totale su **Precision-Recall, F1-Score** e impatto economico.

3. **Specializzazione dei Modelli:** - **XGBoost**: Lavora su dati storici etichettati (pattern noti).

   - **Autoencoders (Deep Learning)**: Cerca errori di ricostruzione statistica per intercettare anomalie **Zero-Day** mai viste prima.

---

## 3. Linee Guida per il Codice (Modulo A & B)

- **Il Wrapper Layer**: Tutta la complessità dei modelli è nascosta dietro l'interfaccia unificata `wrapper.predict_all(dati_json)`. FastAPI parlerà solo con questo wrapper.

- **Pre-processing Atomico**: Ogni modello ha la sua pipeline Scikit-Learn dotata di `ColumnTransformer`. Se l'Autoencoder richiede solo le 5 colonne numeriche normalizzate tra 0 e 1, e XGBoost vuole tutte e 10 le colonne grezze codificate, la pipeline specifica isola e trasforma i dati internamente per evitare il **Data Leakage**.