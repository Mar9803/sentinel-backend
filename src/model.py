from sklearn.ensemble import IsolationForest
import joblib
import os

class FraudDetector:
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        self.is_trained = False

    def predict(self, X):
        if not self.is_trained:
            raise Exception("Il modello deve essere addestrato!")
        return self.model.predict(X)

    def get_scores(self, X):
        return self.model.decision_function(X)

    def load_model(self, filepath):
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            self.is_trained = True
            print(f"📂 Modello caricato da: {filepath}")
        else:
            raise FileNotFoundError(f"⚠️ File {filepath} non trovato")