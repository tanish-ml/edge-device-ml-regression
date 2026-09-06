import pandas as pd
import numpy as np
import json
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
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
        for f2 in TOP_MI[i+1:]:
            X[f"{f1}_{f2}"] = X[f1] * X[f2]
    return X

# 2. OPTUNA OBJECTIVE
def objective(trial):
    # Tightened Preprocessing
    fs_quantile = trial.suggest_float("fs_quantile", 0.93, 0.95)
    pca_components = trial.suggest_int("pca_components", 24, 28)
    
    # Specialized CatBoost Params
    cat_params = {
        "iterations": trial.suggest_int("cat_n", 1200, 2500),
        "depth": trial.suggest_int("cat_depth", 7, 9),
        "learning_rate": trial.suggest_float("cat_lr", 0.01, 0.05, log=True),
        "l2_leaf_reg": trial.suggest_float("cat_l2", 3.0, 15.0),
        "bagging_temperature": trial.suggest_float("cat_bagging", 0.6, 0.9),
        "random_strength": trial.suggest_float("cat_random", 1.0, 3.0),
        "border_count": 254,
        "random_state": 42,
        "verbose": 0
    }
    
    # Specialized LightGBM Params
    lgb_params = {
        "n_estimators": trial.suggest_int("lgb_n", 1000, 1500),
        "num_leaves": trial.suggest_int("lgb_leaves", 25, 45),
        "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.04, log=True),
        "reg_alpha": trial.suggest_float("lgb_alpha", 1.0, 10.0, log=True),
        "reg_lambda": trial.suggest_float("lgb_lambda", 1.0, 10.0, log=True),
        "boosting_type": "gbdt",
        "random_state": 42,
        "verbose": -1
    }

    # Ensemble Weights
    w_lgb = trial.suggest_float("w_lgb", 0, 0.01)
    w_cat = trial.suggest_float("w_cat", 0.85, 1)   

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        # Feature Engineering & PCA
        X_tr_eng = feature_engineering(X_tr)
        X_val_eng = feature_engineering(X_val)

        pca = PCA(n_components=pca_components, random_state=42)
        pca_cols = [f"pca_{i}" for i in range(pca_components)]
        X_tr_eng[pca_cols] = pca.fit_transform(X_tr_eng.filter(regex="feat_"))
        X_val_eng[pca_cols] = pca.transform(X_val_eng.filter(regex="feat_"))

        # Feature Selection (Pilot using LightGBM)
        pilot = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        pilot.fit(X_tr_eng, y_tr)
        imp = pd.Series(pilot.feature_importances_, index=X_tr_eng.columns)
        selected = imp[imp >= imp.quantile(fs_quantile)].index.tolist()
        
        X_tr_final = X_tr_eng[selected]
        X_val_final = X_val_eng[selected]

        # Target Transform
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=42)
        y_tr_t = qt.fit_transform(y_tr.values.reshape(-1, 1)).ravel()

        # Training
        m_lgb = LGBMRegressor(**lgb_params).fit(X_tr_final, y_tr_t)
        m_cat = CatBoostRegressor(**cat_params).fit(X_tr_final, y_tr_t)

        # Blended Prediction
        total_w = w_lgb + w_cat
        p_lgb = m_lgb.predict(X_val_final)
        p_cat = m_cat.predict(X_val_final)
        val_preds_t = (w_lgb * p_lgb + w_cat * p_cat) / total_w
        
        val_preds = qt.inverse_transform(val_preds_t.reshape(-1, 1)).ravel()
        cv_scores.append(r2_score(y_val, val_preds))

    return np.mean(cv_scores)

# 3. RUN STUDY
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50) 

best = study.best_params
print("\n--- Best Duo-Stack Params ---")
print(best)

with open("best_parameters.json", "w") as f:
    json.dump(best, f, indent=4)

# 4. FINAL EVALUATION
X_tr_full = feature_engineering(X_train)
X_te_full = feature_engineering(X_test)

pca = PCA(n_components=best['pca_components'], random_state=42)
pca_cols = [f"pca_{i}" for i in range(best['pca_components'])]
X_tr_full[pca_cols] = pca.fit_transform(X_tr_full.filter(regex="feat_"))
X_te_full[pca_cols] = pca.transform(X_te_full.filter(regex="feat_"))

# Selection (Pilot)
pilot = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
pilot.fit(X_tr_full, y_train)
imp = pd.Series(pilot.feature_importances_, index=X_tr_full.columns)
selected = imp[imp >= imp.quantile(best['fs_quantile'])].index.tolist()

X_tr_f, X_te_f = X_tr_full[selected], X_te_full[selected]

qt = QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=42)
y_tr_t = qt.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Final Models
final_lgb = LGBMRegressor(
    n_estimators=best['lgb_n'], num_leaves=best['lgb_leaves'], 
    learning_rate=best['lgb_lr'], reg_alpha=best['lgb_alpha'], 
    reg_lambda=best['lgb_lambda'], random_state=42, verbose=-1
).fit(X_tr_f, y_tr_t)

final_cat = CatBoostRegressor(
    iterations=best['cat_n'], depth=best['cat_depth'], 
    learning_rate=best['cat_lr'], l2_leaf_reg=best['cat_l2'], 
    bagging_temperature=best['cat_bagging'], random_strength=best['cat_random'],
    random_state=42, verbose=0
).fit(X_tr_f, y_tr_t)

# Final Blend
total_w = best['w_lgb'] + best['w_cat']
test_preds_t = (best['w_lgb'] * final_lgb.predict(X_te_f) + best['w_cat'] * final_cat.predict(X_te_f)) / total_w
test_preds = qt.inverse_transform(test_preds_t.reshape(-1, 1)).ravel()

print("\n--- FINAL HOLDOUT PERFORMANCE ---")
print(f"R2:   {r2_score(y_test, test_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, test_preds)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, test_preds):.4f}")