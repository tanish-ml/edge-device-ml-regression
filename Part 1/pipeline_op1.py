import pandas as pd
import numpy as np
import json
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load data
X_raw = pd.read_csv("dataset_7.csv")
y_raw = pd.read_csv("target_7.csv")["target01"]

# Initial Split
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

TOP_MI = ['feat_112','feat_143','feat_29','feat_240','feat_219','feat_179','feat_259']

def feature_engineering(X):
    X = X.copy()
    for i, f1 in enumerate(TOP_MI):
        for f2 in TOP_MI[i+1:]:
            X[f"{f1}_{f2}"] = X[f1] * X[f2]
    return X

def objective(trial):
    # 1. Suggest Hyperparameters
    fs_quantile = trial.suggest_float("fs_quantile", 0.85, 0.96)
    pca_components = trial.suggest_int("pca_components", 12, 25)
    
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n", 300, 1000),
        "max_depth": trial.suggest_int("xgb_depth", 3, 8),
        "learning_rate": trial.suggest_float("xgb_lr", 0.02, 0.07, log=True),
        "subsample": trial.suggest_float("xgb_sub", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("xgb_lambda", 1e-2, 10.0, log=True), # NEW
    }
    
    lgb_params = {
        "n_estimators": trial.suggest_int("lgb_n", 300, 1000),
        "num_leaves": trial.suggest_int("lgb_leaves", 15, 63),
        "learning_rate": trial.suggest_float("lgb_lr", 0.02, 0.07, log=True),
        "reg_alpha": trial.suggest_float("lgb_alpha", 1e-3, 1.0, log=True), # NEW
    }
    
    cat_params = {
        "iterations": trial.suggest_int("cat_n", 300, 1000),
        "depth": trial.suggest_int("cat_depth", 4, 8),
        "learning_rate": trial.suggest_float("cat_lr", 0.01, 0.1, log=True),
        "verbose": 0,
    }

    # 2. Cross-Validation Loop
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        # Preprocessing
        X_tr = feature_engineering(X_tr)
        X_val = feature_engineering(X_val)

        pca = PCA(n_components=pca_components, random_state=42)
        pca_tr = pca.fit_transform(X_tr.filter(regex="feat_"))
        pca_val = pca.transform(X_val.filter(regex="feat_"))
        for i in range(pca_components):
            X_tr[f"pca_{i}"] = pca_tr[:, i]
            X_val[f"pca_{i}"] = pca_val[:, i]

        # Feature Selection
        pilot = XGBRegressor(n_estimators=100, random_state=42)
        pilot.fit(X_tr, y_tr)
        imp = pd.Series(pilot.feature_importances_, index=X_tr.columns)
        selected = imp[imp >= imp.quantile(fs_quantile)].index.tolist()
        
        X_tr, X_val = X_tr[selected], X_val[selected]

        # Target Transform
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=42)
        y_tr_t = qt.fit_transform(y_tr.values.reshape(-1, 1)).ravel()

        # Training
        m1 = XGBRegressor(**xgb_params, random_state=42)
        m2 = LGBMRegressor(**lgb_params, random_state=42, verbose=-1)
        m3 = CatBoostRegressor(**cat_params, random_state=42)

        m1.fit(X_tr, y_tr_t)
        m2.fit(X_tr, y_tr_t)
        m3.fit(X_tr, y_tr_t)

        # Meta-learner (Simple Average for tuning speed, or use OOF for Ridge)
        meta_X_val = np.column_stack([m1.predict(X_val), m2.predict(X_val), m3.predict(X_val)])
        # Instead of np.mean(meta_X_val, axis=1)
        w1 = trial.suggest_float("w_xgb", 0.0, 1.0)
        w2 = trial.suggest_float("w_lgb", 0.0, 1.0)
        w3 = trial.suggest_float("w_cat", 0.0, 1.0)
        # Normalize weights so they sum to 1
        total_w = w1 + w2 + w3
        val_preds_t = (w1*m1.predict(X_val) + w2*m2.predict(X_val) + w3*m3.predict(X_val)) / total_w
        
        val_preds = qt.inverse_transform(val_preds_t.reshape(-1, 1)).ravel()
        scores.append(r2_score(y_val, val_preds))

    return np.mean(scores)

# Run Study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30) # Increase n_trials for better results

print("Best Parameters:", study.best_params)

# Save parameters to JSON
with open("best_parameters.json", "w") as f:
    json.dump(study.best_params, f, indent=4)

# ======================================================
# FINAL TRAINING WITH BEST PARAMS
# ======================================================
best = study.best_params

X_tr_final = feature_engineering(X_train)
X_te_final = feature_engineering(X_test)

# PCA
pca = PCA(n_components=best['pca_components'], random_state=42)
pca_tr = pca.fit_transform(X_tr_final.filter(regex="feat_"))
pca_te = pca.transform(X_te_final.filter(regex="feat_"))
for i in range(best['pca_components']):
    X_tr_final[f"pca_{i}"] = pca_tr[:, i]
    X_te_final[f"pca_{i}"] = pca_te[:, i]

# Selection
pilot = XGBRegressor(n_estimators=100, random_state=42)
pilot.fit(X_tr_final, y_train)
imp = pd.Series(pilot.feature_importances_, index=X_tr_final.columns)
selected = imp[imp >= imp.quantile(best['fs_quantile'])].index.tolist()

X_tr_final, X_te_final = X_tr_final[selected], X_te_final[selected]

# Final Model
qt = QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=42)
y_tr_t = qt.fit_transform(y_train.values.reshape(-1,1)).ravel()

m1 = XGBRegressor(n_estimators=best['xgb_n'], max_depth=best['xgb_depth'], learning_rate=best['xgb_lr'], subsample=best['xgb_sub'])
m2 = LGBMRegressor(n_estimators=best['lgb_n'], num_leaves=best['lgb_leaves'], learning_rate=best['lgb_lr'], verbose=-1)
m3 = CatBoostRegressor(iterations=best['cat_n'], depth=best['cat_depth'], learning_rate=best['cat_lr'], verbose=0)

m1.fit(X_tr_final, y_tr_t)
m2.fit(X_tr_final, y_tr_t)
m3.fit(X_tr_final, y_tr_t)

# Ensemble Prediction
meta_X_te = np.column_stack([m1.predict(X_te_final), m2.predict(X_te_final), m3.predict(X_te_final)])
test_preds_t = np.mean(meta_X_te, axis=1)
test_preds = qt.inverse_transform(test_preds_t.reshape(-1, 1)).ravel()

print(f"\nFINAL TEST R2: {r2_score(y_test, test_preds):.4f}")   