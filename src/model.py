from sklearn.ensemble import IsolationForest
import joblib # libreria per salvare il modello una volta addestrato
import os


class FraudDetector:
    """
    Wrapper per il modello di Machine Learning (Isolation Forest).
    Responsabile dell'addestramento e della predizione delle anomalie.
    Incapsula le operaizoni: train, predict, scores.
    """
    def __init__(self, contamination=0.01):
        # contamination è la percentuale attesa di frodi (es. 1%)
        self.model = IsolationForest(
            contamination=contamination, 
            random_state = 42,
            n_jobs=-1 # Usa tutti i core della CPU per la velocità
        )
        self.is_trained = False

    def train(self, X):
        """Addestra il modello sulle feature (grezze + grafi)."""
        print("🧠 Addestramento Isolation Forest in corso...")
        self.model.fit(X)
        self.is_trained = True
        print("✅ Addestramento completato.")

    def predict(self, X):
        """
        Predice se una transazione è un'anomalia.
        Ritorna: 1 per normale, -1 per anomalia (standard Scikit-Learn)
        """
        if not self.is_trained:
            raise Exception("Il modello deve essere addestrato prima della predizione!")
        return self.model.predict(X)
    
    def get_scores(self, X):
        """Restituisce il punteggio di anomalia (più è basso, più è sospetto)."""
        return self.model.decision_function(X)

# metodi save_model,  load_model --> 
   
    def save_model(self, filepath='models/sentinel_model.pkl'):
        """Salva il modello su disco."""
        # Crea la cartella models se non esiste
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"💾 Modello salvato con successo in: {filepath}")

    def load_model(self, filepath='models/sentinel_model.pkl'):
        """Carica il modello dal disco."""
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            self.is_trained = True
            print(f"📂 Modello caricato da: {filepath}")
        else:
            print(f"⚠️ Nessun modello trovato in {filepath}")