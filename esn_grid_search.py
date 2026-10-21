import time
import numpy as np
import pandas as pd
from aeon.datasets import load_classification
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import ParameterGrid
from reservoirpy.nodes import Reservoir

# ==============================================================================
# 1. CARICAMENTO DATASET TRACE (AEON)
# ==============================================================================

# Caricamento del dataset Trace
X_train_raw, y_train_raw = load_classification("Trace", split="train")
X_test_raw, y_test_raw = load_classification("Trace", split="test")

# Trace ha etichette originali "1", "2", "3", "4". 
# Sottraiamo 1 per avere interi a partire da 0 (0, 1, 2, 3) come preferisce scikit-learn
y_train = y_train_raw.astype(int) - 1
y_test = y_test_raw.astype(int) - 1

# ReservoirPy si aspetta ogni serie temporale con shape (timesteps, features)
X_train_list = [series[0].reshape(-1, 1) for series in X_train_raw]
X_test_list = [series[0].reshape(-1, 1) for series in X_test_raw]

print(f" -> Campioni di Train: {len(X_train_list)} (lunghezza serie: {X_train_list[0].shape[0]})")
print(f" -> Campioni di Test:  {len(X_test_list)}")


# ==============================================================================
# FUNZIONE HELPER: Estrazione dello Stato del Reservoir
# ==============================================================================
def extract_reservoir_states(reservoir_node, series_list):
    """
    Esegue la rete su ciascuna serie temporale e restituisce la matrice 
    degli stati dei neuroni all'ULTIMO timestep di ogni sequenza (shape: [N_campioni, UNITS]).
    """
    states = []
    for series in series_list:
        res_states = reservoir_node.run(series)
        states.append(res_states[-1, :])  # Stato all'ultimo timestep
    return np.array(states)


# ==============================================================================
# 2. DEFINIZIONE DELLA GRID SEARCH E DEI VALORI DI LAMBDA (ALPHA)
# ==============================================================================
# Griglia di iperparametri per il Reservoir da esplorare in modo esaustivo
param_grid = {
    'units': [100, 200, 300],          # Dimensione dello stato del reservoir
    'sr': [0.7, 0.9, 0.99],            # Spectral Radius (memoria a lungo termine)
    'lr': [0.1, 0.3, 0.5]              # Leaking Rate (velocità di aggiornamento)
}

# Griglia logaritmica di Lambda (Alpha in scikit-learn) che RidgeClassifierCV testerà da solo
alphas_to_test = np.logspace(-3, 5, 10)  # [0.001, 0.007, ..., 100000.0]

grid = ParameterGrid(param_grid)
print(f"\n2. Inizio Grid Search su {len(grid)} combinazioni di iperparametri Reservoir...")

# ==============================================================================
# 3. CICLO ITERATIVO DI MODEL SELECTION (GRID SEARCH + CROSS-VALIDATION)
# ==============================================================================
results_history = []
best_cv_score = -1.0
best_config = None
best_model = None
best_reservoir = None

start_grid_time = time.time()

for i, params in enumerate(grid, 1):
    t_start = time.time()
    
    # 1. Inizializzazione del Reservoir con la combinazione corrente
    res = Reservoir(
        units=params['units'],
        sr=params['sr'],
        lr=params['lr'],
        rc_connectivity=0.1,
        seed=42
    )
    
    # 2. Estrazione delle feature (stati dei neuroni) sui dati di Train
    X_train_states = extract_reservoir_states(res, X_train_list)
    
    # 3. Model Selection Automatica su Lambda tramite 5-Fold Cross-Validation
    clf = RidgeClassifierCV(alphas=alphas_to_test, cv=5)
    clf.fit(X_train_states, y_train)
    
    # Punteggio di accuratezza media ottenuto in Cross-Validation e miglior Lambda trovato
    cv_score = clf.best_score_
    selected_lambda = clf.alpha_
    elapsed_time = time.time() - t_start
    
    # Salva la traccia nei risultati
    results_history.append({
        'run': i,
        'units': params['units'],
        'sr': params['sr'],
        'lr': params['lr'],
        'best_lambda': selected_lambda,
        'cv_accuracy': cv_score,
        'time_sec': elapsed_time
    })
    
    print(f"[{i:02d}/{len(grid)}] units={params['units']:3d} | sr={params['sr']:.2f} | lr={params['lr']:.2f} | "
          f"Lambda CV: {selected_lambda:10.5f} | Accuracy CV: {cv_score*100:6.2f}% ({elapsed_time:.1f}s)")
    
    # Mantiene la migliore configurazione in base alla Cross-Validation
    if cv_score > best_cv_score:
        best_cv_score = cv_score
        best_config = params
        best_model = clf
        best_reservoir = res

grid_total_time = time.time() - start_grid_time
print(f"\nGrid Search completata in {grid_total_time:.2f} secondi!")

# Converti lo storico in un DataFrame Pandas per eventuale analisi o salvataggio
df_results = pd.DataFrame(results_history)

# ==============================================================================
# 4. VALUTAZIONE FINALE E METRICHE SUL TEST SET INEDITO
# ==============================================================================
print("\n" + "=" * 70)
print("3. RISULTATI DELLA MODEL SELECTION & VALUTAZIONE FINALE SUL TEST SET")
print("=" * 70)
print(f"Migliori Iperparametri Reservoir : {best_config}")
print(f"Miglior Lambda (Readout Ridge)   : {best_model.alpha_:.5f}")
print(f"Migliore Cross-Validation Acc   : {best_cv_score * 100:.2f}%")
print("-" * 70)

# 1. Estrai gli stati per il Test Set usando il Reservoir vincente
X_test_states = extract_reservoir_states(best_reservoir, X_test_list)

# 2. Fai la predizione usando il modello Ridge ottimizzato
start_test_time = time.time()
y_pred = best_model.predict(X_test_states)
test_inference_time = time.time() - start_test_time

# 3. Calcolo dell'Accuracy Finale sul Test Set
test_acc = accuracy_score(y_test, y_pred)
print(f"TEST ACCURACY DEFINITIVA        : {test_acc * 100:.2f}%")
print(f"Tempo di Inferenza sul test set : {test_inference_time:.4f} s")
print("-" * 70)

# 4. Report di Classificazione per Classe
# Trace ha 4 classi (associate a diversi pattern di guasto/monitoraggio strumentale)
class_names = ["Classe 0", "Classe 1", "Classe 2", "Classe 3"]
print("\nREPORT DI CLASSIFICAZIONE (Per Classe):")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

# 5. Matrice di Confusione
print("MATRICE DI CONFUSIONE:")
print(confusion_matrix(y_test, y_pred))

# Opzionale: salva la tabella dei risultati della Grid Search in CSV per la Tesi
# df_results.to_csv("esn_trace_gridsearch_results.csv", index=False)