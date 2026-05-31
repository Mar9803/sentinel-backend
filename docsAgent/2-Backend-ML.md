# Sentinel: Real-Time Fraud Detection Ecosystem

## Documento 2: Specifiche Tecniche Modelli & Logica di Business (Modulo A)

Questo documento istruisce l'Agente sulle specifiche matematiche, di pre-processing e di valutazione dei tre motori di decisione di Sentinel.

---

## 1. Il Primo Muro: Il Motore a Regole Statiche (Euristiche)

Prima di toccare l'intelligenza artificiale, la transazione passa per un sistema di regole fisse. Serve a intercettare le frodi grossolane a costo computazionale vicino allo zero.

- **Obiettivo**: Scrematura iniziale e baseline di confronto per dimostrare il risparmio di efficienza.

- **Regole Standard da Implementare**:

  1. *Velocity Check*: Più di 3 transazioni distinte negli ultimi 60 secondi dallo stesso User ID.

  2. *Impossible Travel*: Transazioni consecutive da due nazioni diverse in un arco temporale inferiore a quello del volo aereo (es. Italia -> USA in 15 minuti).

  3. *High-Risk Amount Threshold*: Importo singola transazione superiore a 3 deviazioni standard rispetto alla media storica dell'utente.

---

## 2. Il Secondo Muro: XGBoost (Supervisionato - Pattern Noti)

Lavora sui dati storici etichettati `target: 0` = legittima, `1` = frode). Impara le impronte digitali delle frodi già avvenute e catalogate nel database.

- **Sensibilità dell'Input**: Richiede dati tabulari completi. Gestisce nativamente i valori nulli, ma le feature categoriche (es. tipo di carta, paese) devono essere codificate (Target Encoding o One-Hot Encoding tramite la pipeline).

- **Gestione dell'Estremo Sbilanciamento**: 

  Nelle frodi, il dataset ha solitamente il 99.5% di transazioni sane e lo 0.5% di frodi. Per evitare che XGBoost impari a dire sempre "legittima" per avere un'accuratezza del 99.5%, useremo il parametro `scale_pos_weight` calcolato come:

  $$\text{scale\_pos\_weight} = \frac{\text{Totale Classi Negative (Sane)}}{\text{Totale Classi Positive (Frodi)}}$$

---

## 3. Il Terzo Muro: Autoencoder & Isolation Forest (Non Supervisionato - Anomalie Zero-Day)

Questo layer non cerca pattern noti, ma cerca tutto ciò che si discosta dal comportamento "normale". È fondamentale per le frodi **Zero-Day** (tecniche d'attacco mai viste prima).

### Isolation Forest (Baseline Statistica)

- **Logica**: Isola le anomalie tramite partizionamento casuale delle feature. Le anomalie richiedono meno split per essere isolate rispetto ai dati normali.

### Autoencoder (Deep Learning - PyTorch/Keras)

- **Logica**: Una rete neurale a "imbuto" (Encoder-Decoder) addestrata **esclusivamente su transazioni legittime**. 

- **Pre-processing**: Sensibilissimo alla scala dei dati. Richiede input esclusivamente numerici e normalizzati rigorosamente tra 0 e 1 (tramite `MinMaxScaler` o `StandardScaler` isolati nella sua pipeline).

- **Calcolo dello Score di Frode**: 

  La rete prende l'input $X$, lo comprime e prova a ricostruirlo come $\hat{X}$. Se l'errore di ricostruzione (Mean Squared Error) supera una determinata soglia $\tau$, la transazione è considerata un'anomalia:

  $$\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} (X_i - \hat{X}_i)^2 > \tau$$

---

## 4. Il Wrapper Layer `.predict_all()`) e Logica di Decisione

Il file `wrapper.py` riceve il JSON grezzo della transazione e orchestrerà le chiamate in parallelo:

```text

               [ Input JSON Transazione ]

                           │

                           ▼

               ┌───────────────────────┐

               │    Regole Statiche    │ ──(Se triggerata flag Critica)──> [ BLOCCO IMMEDIATO ]

               └───────────────────────┘

                           │

                           ▼

               ┌───────────────────────┐

       ┌───────│    Wrapper Layer      │───────┐

       │       └───────────────────────┘       │

       ▼                   ▼                   ▼

┌──────────────┐    ┌──────────────┐    ┌──────────────┐

│Pipeline XGB  │    │Pipeline I.F. │    │Pipeline A.E. │

└──────────────┘    └──────────────┘    └──────────────┘

       │                   │                   │

  (Score 0-1)         (Score 0-1)         (Score 0-1)

       │                   │                   │

       └───────────────────┼───────────────────┘

                           ▼

               ┌───────────────────────┐

               │     Decision Logic    │ ──> Score Finale Pesato & SHAP

               └───────────────────────┘