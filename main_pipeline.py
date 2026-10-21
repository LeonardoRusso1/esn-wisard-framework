import time
import pickle
import numpy as np
import pandas as pd
from aeon.datasets import load_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from reservoirpy.nodes import Reservoir
from wisard.classifier import WisardClassifier

BEST_ESN_PER_DATASET = {
    "ECG5000":  {'units': 300, 'sr': 0.99, 'lr': 0.3, 'rc_connectivity': 0.1, 'seed': 42},
    "Wafer":    {'units': 300, 'sr': 0.99, 'lr': 0.5, 'rc_connectivity': 0.1, 'seed': 42},
    "GunPoint": {'units': 200, 'sr': 0.99, 'lr': 0.1, 'rc_connectivity': 0.1, 'seed': 42},
    "FordA":    {'units': 100, 'sr': 0.99, 'lr': 0.5, 'rc_connectivity': 0.1, 'seed': 42},
    "Yoga":     {'units': 300, 'sr': 0.99, 'lr': 0.3, 'rc_connectivity': 0.1, 'seed': 42},
    "SyntheticControl": {'units': 200, 'sr': 0.99, 'lr': 0.3, 'rc_connectivity': 0.1, 'seed': 42},
    "ItalyPowerDemand": {'units': 300, 'sr': 0.7,  'lr': 0.3, 'rc_connectivity': 0.1, 'seed': 42},
    "Trace":            {'units': 200, 'sr': 0.99, 'lr': 0.1, 'rc_connectivity': 0.1, 'seed': 42}
}

POOLING_PER_DATASET = {
    "ECG5000":  "multi",
    "Wafer":    "multi",
    "GunPoint": "multi",
    "Yoga":     "multi",
    "FordA":    "multi",
    "SyntheticControl": "multi",
    "ItalyPowerDemand": "multi",
    "Trace":            "multi"
}

ALPHAS_RIDGE = np.logspace(-3, 5, 10)

PARAM_GRID_WISARD = {
    'n_bits': [32],
    'n_tics': [16, 32, 64],
    'n_quantiles': [100, 200, 300, 500]
}

def extract_reservoir_states(res_node, series_list, pooling='last'):
    states = []
    for series in series_list:
        try:
            res_node.reset()
        except AttributeError:
            pass
        res_states = res_node.run(series)
        
        if pooling == 'mean':
            states.append(np.mean(res_states, axis=0))
        elif pooling == 'max':
            states.append(np.max(res_states, axis=0))
        elif pooling == 'multi':
            states.append(np.hstack([np.mean(res_states, axis=0), np.max(res_states, axis=0), np.std(res_states, axis=0)]))
        else:
            states.append(res_states[-1, :])
            
    return np.array(states)

def grid_search_minmax(X_tr_states, y_tr, sss):
    best_score, best_params = -1.0, {'n_bits': 32, 'n_tics': 16}
    for bits in PARAM_GRID_WISARD['n_bits']:
        for tics in PARAM_GRID_WISARD['n_tics']:
            scores = []
            for tr_idx, val_idx in sss.split(X_tr_states, y_tr):
                scaler = MinMaxScaler()
                X_tr_fold = scaler.fit_transform(X_tr_states[tr_idx])
                X_val_fold = scaler.transform(X_tr_states[val_idx])
                clf = WisardClassifier(n_bits=bits, n_tics=tics, bleaching=True).fit(X_tr_fold, y_tr[tr_idx])
                scores.append(f1_score(y_tr[val_idx], clf.predict(X_val_fold), average='macro'))
            if np.mean(scores) > best_score:
                best_score, best_params = np.mean(scores), {'n_bits': bits, 'n_tics': tics}
    return best_params

def grid_search_quantile(X_tr_states, y_tr, distribution, sss):
    best_score = -1.0
    best_params = {'n_bits': 32, 'n_tics': 16, 'n_quantiles': 100}
    valid_quantiles = [q for q in PARAM_GRID_WISARD['n_quantiles'] if q <= len(X_tr_states)] or [len(X_tr_states)]
        
    for n_q in valid_quantiles:
        for bits in PARAM_GRID_WISARD['n_bits']:
            for tics in PARAM_GRID_WISARD['n_tics']:
                scores = []
                for tr_idx, val_idx in sss.split(X_tr_states, y_tr):
                    pipe = make_pipeline(StandardScaler(), QuantileTransformer(n_quantiles=min(n_q, len(tr_idx)), output_distribution=distribution, random_state=42))
                    X_tr_fold = pipe.fit_transform(X_tr_states[tr_idx])
                    X_val_fold = pipe.transform(X_tr_states[val_idx])
                    clf = WisardClassifier(n_bits=bits, n_tics=tics, bleaching=True).fit(X_tr_fold, y_tr[tr_idx])
                    scores.append(f1_score(y_tr[val_idx], clf.predict(X_val_fold), average='macro'))
                if np.mean(scores) > best_score:
                    best_score, best_params = np.mean(scores), {'n_bits': bits, 'n_tics': tics, 'n_quantiles': n_q}
    return best_params

def get_model_size_kb(model):
    try:
        return len(pickle.dumps(model)) / 1024
    except:
        return 0.0

all_results = []

for ds, esn_params in BEST_ESN_PER_DATASET.items():
    pooling_type = POOLING_PER_DATASET[ds]
    print(f"\n==================== DATASET: {ds} (Pooling: {pooling_type.upper()}) ====================")
    
    X_tr_raw, y_tr_raw = load_classification(ds, split="train")
    X_te_raw, y_te_raw = load_classification(ds, split="test")
    
    le = LabelEncoder()
    y_tr, y_te = le.fit_transform(y_tr_raw), le.transform(y_te_raw)
    
    N_classes = len(np.unique(y_tr))
    
    X_tr_list = [series.squeeze().reshape(-1, 1) for series in X_tr_raw]
    X_te_list = [series.squeeze().reshape(-1, 1) for series in X_te_raw]
    
    reservoir = Reservoir(**esn_params)
    X_tr_states = extract_reservoir_states(reservoir, X_tr_list, pooling=pooling_type)
    X_te_states = extract_reservoir_states(reservoir, X_te_list, pooling=pooling_type)
    
    N_features = X_tr_states.shape[1]
    n_test_samples = len(X_te_states)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    # 1. Baseline RidgeCV
    t0 = time.perf_counter()
    ridge = RidgeClassifierCV(alphas=ALPHAS_RIDGE, cv=5).fit(X_tr_states, y_tr)
    t_fit_ridge = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    y_pred_ridge = ridge.predict(X_te_states)
    t_inf_ridge_ms = ((time.perf_counter() - t0) / n_test_samples) * 1000
    
    ridge_mem_kb = (ridge.coef_.nbytes + getattr(ridge, 'intercept_', np.array([0])).nbytes) / 1024
    ridge_flops = 2 * N_classes * N_features
    
    # 2. WiSARD MinMax
    p_mm = grid_search_minmax(X_tr_states, y_tr, sss)
    scaler_mm = MinMaxScaler()
    X_tr_mm, X_te_mm = scaler_mm.fit_transform(X_tr_states), scaler_mm.transform(X_te_states)
    
    t0 = time.perf_counter()
    wc_mm = WisardClassifier(n_bits=p_mm['n_bits'], n_tics=p_mm['n_tics'], bleaching=True).fit(X_tr_mm, y_tr)
    t_fit_mm = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    y_pred_mm = wc_mm.predict(X_te_mm)
    t_inf_mm_ms = ((time.perf_counter() - t0) / n_test_samples) * 1000
    
    # 3. WiSARD Gaussiana
    p_gauss = grid_search_quantile(X_tr_states, y_tr, 'normal', sss)
    pipe_g = make_pipeline(StandardScaler(), QuantileTransformer(n_quantiles=p_gauss['n_quantiles'], output_distribution='normal', random_state=42))
    X_tr_g, X_te_g = pipe_g.fit_transform(X_tr_states), pipe_g.transform(X_te_states)
    
    t0 = time.perf_counter()
    wc_g = WisardClassifier(n_bits=p_gauss['n_bits'], n_tics=p_gauss['n_tics'], bleaching=True).fit(X_tr_g, y_tr)
    t_fit_g = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    y_pred_g = wc_g.predict(X_te_g)
    t_inf_g_ms = ((time.perf_counter() - t0) / n_test_samples) * 1000
    
    # 4. WiSARD Quantile Uniforme
    p_quant = grid_search_quantile(X_tr_states, y_tr, 'uniform', sss)
    pipe_u = make_pipeline(StandardScaler(), QuantileTransformer(n_quantiles=p_quant['n_quantiles'], output_distribution='uniform', random_state=42))
    X_tr_u, X_te_u = pipe_u.fit_transform(X_tr_states), pipe_u.transform(X_te_states)
    
    t0 = time.perf_counter()
    wc_u = WisardClassifier(n_bits=p_quant['n_bits'], n_tics=p_quant['n_tics'], bleaching=True).fit(X_tr_u, y_tr)
    t_fit_u = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    y_pred_u = wc_u.predict(X_te_u)
    t_inf_u_ms = ((time.perf_counter() - t0) / n_test_samples) * 1000
    
    models = [
        (f"1. Baseline RidgeCV", y_pred_ridge, t_fit_ridge, t_inf_ridge_ms, ridge_mem_kb, ridge_flops),
        (f"2. WiSARD MinMax", y_pred_mm, t_fit_mm, t_inf_mm_ms, get_model_size_kb(wc_mm), 0),
        (f"3. WiSARD Gaussiana", y_pred_g, t_fit_g, t_inf_g_ms, get_model_size_kb(wc_g), 0),
        (f"4. WiSARD Quantile Uniforme", y_pred_u, t_fit_u, t_inf_u_ms, get_model_size_kb(wc_u), 0)
    ]
    
    for name, pred, t_fit, t_inf, mem_kb, flops in models:
        all_results.append({
            "Dataset": ds, 
            "Modello": name,
            "Acc (%)": round(accuracy_score(y_te, pred) * 100, 2),
            "F1 (%)": round(f1_score(y_te, pred, average='macro') * 100, 2),
            "Fit Time (s)": round(t_fit, 3),
            "Inf Latency (ms)": round(t_inf, 4),
            "Mem (KB)": round(mem_kb, 2),
            "FLOPs": flops
        })

df_final = pd.DataFrame(all_results)
print("\n" + "="*120 + "\n TABELLA FINALE COMPARATIVA (INCLUSE METRICHE HARDWARE/TEMPO)\n" + "="*120)
print(df_final.to_string(index=False))