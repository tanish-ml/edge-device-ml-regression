import pandas as pd
import numpy as np
import json
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor

# 1. DATA LOADING
X_raw = pd.read_csv("dataset_7.csv")
y_raw = pd.read_csv("target_7.csv")["target01"]

# Initial Split
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

TOP_MI = ['feat_112','feat_143','feat_29','feat_240','feat_219','feat_179','feat_259']

def feature_engineering(X):
    X = X.copy()
    for i, f1 in enumerate(TOP_MI):
        # Explicit Polynomial Terms (The 0.92+ Secret Sauce)
        X[f"{f1}_sq"] = X[f1] ** 2
        for f2 in TOP_MI[i+1:]:
            X[f"{f1}_{f2}"] = X[f1] * X[f2]
    return X

# 2. OPTUNA OBJECTIVE (Pure CatBoost)
def objective(trial):
    # Preprocessing ranges based on your successful 0.90+ runs
    fs_quantile = trial.suggest_float("fs_quantile", 0.93, 0.96)
    pca_components = trial.suggest_int("pca_components", 24, 28)
    
    # Deep CatBoost Tuning
    params = {
        "iterations": trial.suggest_int("iterations", 1500, 3000),
        "depth": trial.suggest_int("depth", 7, 10), # Expanded to 10
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5.0, 20.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.5, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1.0, 5.0),
        "od_type": "Iter",
        "od_wait": 50,
        "border_count": 254,
        "bootstrap_type": "Bayesian",
        "random_state": 42,
        "verbose": 0
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        # Process
        X_tr_eng = feature_engineering(X_tr)
        X_val_eng = feature_engineering(X_val)

        pca = PCA(n_components=pca_components, random_state=42)
        pca_cols = [f"pca_{i}" for i in range(pca_components)]
        X_tr_eng[pca_cols] = pca.fit_transform(X_tr_eng.filter(regex="feat_"))
        X_val_eng[pca_cols] = pca.transform(X_val_eng.filter(regex="feat_"))

        # Feature Selection (Pilot using CatBoost for consistency)
        pilot = CatBoostRegressor(iterations=100, depth=6, verbose=0, random_state=42)
        pilot.fit(X_tr_eng, y_tr)
        imp = pd.Series(pilot.get_feature_importance(), index=X_tr_eng.columns)
        selected = imp[imp >= imp.quantile(fs_quantile)].index.tolist()
        
        X_tr_f = X_tr_eng[selected]
        X_val_f = X_val_eng[selected]

        # Target Transform
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=42)
        y_tr_t = qt.fit_transform(y_tr.values.reshape(-1, 1)).ravel()

        # Train & Predict
        model = CatBoostRegressor(**params).fit(X_tr_f, y_tr_t)
        val_preds_t = model.predict(X_val_f)
        
        val_preds = qt.inverse_transform(val_preds_t.reshape(-1, 1)).ravel()
        cv_scores.append(r2_score(y_val, val_preds))

    return np.mean(cv_scores)

# 3. RUN STUDY
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best = study.best_params
print("\n--- Best Pure CatBoost Params ---")
print(best)

with open("best_cat_parameters.json", "w") as f:
    json.dump(best, f, indent=4)

# 4. FINAL EVALUATION
X_tr_full = feature_engineering(X_train)
X_te_full = feature_engineering(X_test)

pca = PCA(n_components=best['pca_components'], random_state=42)
pca_cols = [f"pca_{i}" for i in range(best['pca_components'])]
X_tr_full[pca_cols] = pca.fit_transform(X_tr_full.filter(regex="feat_"))
X_te_full[pca_cols] = pca.transform(X_te_full.filter(regex="feat_"))

# Re-run Pilot for Final Selection
pilot = CatBoostRegressor(iterations=100, depth=6, verbose=0, random_state=42)
pilot.fit(X_tr_full, y_train)
imp = pd.Series(pilot.get_feature_importance(), index=X_tr_full.columns)
selected = imp[imp >= imp.quantile(best['fs_quantile'])].index.tolist()

X_tr_fin, X_te_fin = X_tr_full[selected], X_te_full[selected]

qt = QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=42)
y_tr_t = qt.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Final sovereign model
final_model = CatBoostRegressor(
    iterations=best['iterations'],
    depth=best['depth'],
    learning_rate=best['learning_rate'],
    l2_leaf_reg=best['l2_leaf_reg'],
    bagging_temperature=best['bagging_temperature'],
    random_strength=best['random_strength'],
    border_count=254,
    random_state=42,
    verbose=0
).fit(X_tr_fin, y_tr_t)

test_preds_t = final_model.predict(X_te_fin)
test_preds = qt.inverse_transform(test_preds_t.reshape(-1, 1)).ravel()

print("\n--- FINAL SOVEREIGN PERFORMANCE ---")
print(f"R2:   {r2_score(y_test, test_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, test_preds)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, test_preds):.4f}")