# Sentinel: Real-Time Fraud Detection Ecosystem

## Documento 3: Specifiche Frontend & Storytelling degli Scenari (Modulo C)

Questo documento definisce l'interfaccia utente (Astro + FastHTML/Streamlit) e la logica degli scenari interattivi "One-Click Simulation".

---

## 1. Il Cuore dell'Esperienza: One-Click Scenario Simulation

Invece di obbligare l'utente a caricare dataset complessi, la dashboard offrirà 3 pulsanti principali. Ogni pulsante simula l'invio di un cluster di transazioni (un attacco hacker specifico) per dimostrare visivamente la specializzazione dei modelli.

### ──────── SCENARIO A: "L'Attacco dell'Ingegnere Finanziario" ────────

* **Cosa simula**: Un frodatore esperto che ha studiato lo storico. Fa transazioni che imitano i pattern passati ma con importi leggermente diversi.

* **Comportamento del Sistema**: 

  - Il motore a Regole fallisce (i limiti non vengono superati).

  - L'Autoencoder tentenna (l'errore di ricostruzione è medio).

  - **XGBoost fa centro!** Identifica i pattern correlati (es. combinazione specifica di orario, tipo di merchant e paese).

* **Messaggio per l'utente**: *XGBoost è imbattibile sui pattern storici noti e catalogati.*

### ──────── SCENARIO B: "L'Attacco Lampo (Zero-Day Attack)" ────────

* **Cosa simula**: Un attacco hacker coordinato totalmente nuovo. Nessuno ha mai visto questo comportamento, le transazioni sembrano lecite ma la sequenza e la distribuzione statistica delle feature latenti sono anomale.

* **Comportamento del Sistema**:

  - Le Regole falliscono.

  - XGBoost fallisce (non ha mai visto queste frodi nel passato, quindi per lui sono transazioni sane).

  - **L'Autoencoder (Deep Learning) fa centro!** Registra un errore di ricostruzione altissimo ($MSE > \tau$).

* **Messaggio per l'utente**: *Qui crolla il mito del machine learning classico. Senza un modello non supervisionato come l'Autoencoder, questa frode avrebbe violato il sistema.*

### ──────── SCENARIO C: "La Frode Grossolana (Il Dilettante)" ────────

* **Cosa simula**: Un utente che prova a usare una carta clonata facendo 5 transazioni in 10 secondi dall'Asia, mentre il vero proprietario ha pagato a Roma 5 minuti prima.

* **Comportamento del Sistema**:

  - **Le Regole Statiche bloccano tutto all'istante** (Velocity Check & Impossible Travel).

  - Il motore di Machine Learning viene spento/bypassato per questa transazione (risparmio di risorse hardware).

* **Messaggio per l'utente**: *Pragmatismo ingegneristico: non serve consumare GPU o fare inferenze complesse per bloccare una frode evidente. Le regole fisse fanno risparmiare migliaia di euro in computazione.*

---

## 2. Flusso Tecnico della Simulazione

1. L'utente clicca sul bottone dello **Scenario B** nella dashboard in Astro.

2. Il frontend invia un payload JSON simulato (o una serie di richieste) all'endpoint `/predict` di FastAPI.

3. FastAPI risponde con i singoli score dei modelli `XGB: 0.12`, `AE: 0.94`, `Rules: 0`).

4. Astro riceve la risposta e aggiorna i grafici in tempo reale, illuminando visivamente di **VERDE** il modello vincente (l'Autoencoder) e di **ROSSO** quelli che hanno fallito, mostrando il grafico di spiegabilità (SHAP).