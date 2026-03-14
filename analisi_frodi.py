import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Caricamento Dati
print("Caricamento in corso...")
df = pd.read_csv('creditcard.csv')

# 2. Roadmap 2026: Feature Scaling
# 'Time' e 'Amount' hanno scale troppo diverse dagli altri. Li normalizziamo.
scaler = StandardScaler()

df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Rimuoviamo le colonne originali non scalate
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# 3. Analisi dello sbilanciamento (Imbalance)
print(f"\nDistribuzione Classi:\n{df['Class'].value_counts()}")
print(f"\nPercentuale Frodi: {df['Class'].value_counts(normalize=True)[1]*100:.4f}%")