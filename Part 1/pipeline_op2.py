import pandas as pd
import numpy as np
import json
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# 1. DATA LOADING
X_raw = pd.read_csv("dataset_7.csv")
y_raw = pd.read_csv("target_7.csv")["target01"]

# Initial Split - Keep the test set totally separate
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

TOP_MI = ['feat_112','feat_143','feat_29','feat_240','feat_219','feat_179','feat_259']

def feature_engineering(X):
    X = X.copy()
    for i, f1 in enumerate(TOP_MI):
        for f2 in TOP_MI[i+1:]:
            X[f"{f1}_{f2}"] = X[f1] * X[f2]
    return X

# # 2. OPTUNA OBJECTIVE
def objective(trial):
    # Suggest Preprocessing Params
    fs_quantile = trial.suggest_float("fs_quantile", 0.88, 0.95)
    pca_components = trial.suggest_int("pca_components", 20, 28)
    
    # Suggest Model Params (Expanded with Regularization)
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n", 600, 1500),
        "max_depth": trial.suggest_int("xgb_depth", 3, 5),
        "learning_rate": trial.suggest_float("xgb_lr", 0.005, 0.05, log=True),
        "subsample": trial.suggest_float("xgb_sub", 0.6, 0.9),
        "reg_lambda": trial.suggest_float("xgb_lambda", 1e-2, 20.0, log=True),
        "random_state": 42
    }
    
    lgb_params = {
        "n_estimators": trial.suggest_int("lgb_n", 600, 1500),
        "num_leaves": trial.suggest_int("lgb_leaves", 20, 35),
        "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.05, log=True),
        "reg_alpha": trial.suggest_float("lgb_alpha", 1e-3, 10.0, log=True),
        "boosting_type": trial.suggest_categorical("lgb_type", ["gbdt", "dart"]), # NEW
        "random_state": 42,
        "verbose": -1
    }
    
    cat_params = {
        "iterations": trial.suggest_int("cat_n", 600, 1500),
        "depth": trial.suggest_int("cat_depth", 4, 8),
        "learning_rate": trial.suggest_float("cat_lr", 0.005, 0.04, log=True),
        "l2_leaf_reg": trial.suggest_float("cat_l2", 5.0, 25.0),
        "bagging_temperature": trial.suggest_float("cat_bagging", 0.0, 1.0), # NEW
        "random_strength": trial.suggest_float("cat_random", 1.0, 10.0), # NEW
        "border_count": 254, # Maximum precision for numerical features
        "random_state": 42,
        "verbose": 0
    }

    # Suggest Weights for Blending
    w_xgb = trial.suggest_float("w_xgb", 0.0, 0.3)
    w_lgb = trial.suggest_float("w_lgb", 0.0, 0.5)
    w_cat = trial.suggest_float("w_cat", 0.0, 1.0)

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

        # Feature Selection (Pilot)
        pilot = XGBRegressor(n_estimators=100, random_state=42)
        pilot.fit(X_tr_eng, y_tr)
        imp = pd.Series(pilot.feature_importances_, index=X_tr_eng.columns)
        selected = imp[imp >= imp.quantile(fs_quantile)].index.tolist()
        
        X_tr_final = X_tr_eng[selected]
        X_val_final = X_val_eng[selected]

        # Target Transform
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=42)
        y_tr_t = qt.fit_transform(y_tr.values.reshape(-1, 1)).ravel()

        # Fit Models
        m1 = XGBRegressor(**xgb_params).fit(X_tr_final, y_tr_t)
        m2 = LGBMRegressor(**lgb_params).fit(X_tr_final, y_tr_t)
        m3 = CatBoostRegressor(**cat_params).fit(X_tr_final, y_tr_t)

        # Weighted Prediction
        total_w = w_xgb + w_lgb + w_cat
        p1, p2, p3 = m1.predict(X_val_final), m2.predict(X_val_final), m3.predict(X_val_final)
        val_preds_t = (w_xgb*p1 + w_lgb*p2 + w_cat*p3) / total_w
        
        # Inverse Transform & Score
        val_preds = qt.inverse_transform(val_preds_t.reshape(-1, 1)).ravel()
        cv_scores.append(r2_score(y_val, val_preds))

    return np.mean(cv_scores)



# 3. RUN STUDY
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50) 

best = study.best_params
print("\n--- Best Params Found ---")
print(best)

with open("best_parameters.json", "w") as f:
    json.dump(best, f, indent=4)

# 4. FINAL EVALUATION ON HOLDOUT
X_tr_full = feature_engineering(X_train)
X_te_full = feature_engineering(X_test)

# PCA
pca = PCA(n_components=best['pca_components'], random_state=42)
pca_cols = [f"pca_{i}" for i in range(best['pca_components'])]
X_tr_full[pca_cols] = pca.fit_transform(X_tr_full.filter(regex="feat_"))
X_te_full[pca_cols] = pca.transform(X_te_full.filter(regex="feat_"))

# Feature Selection
pilot = XGBRegressor(n_estimators=100, random_state=42)
pilot.fit(X_tr_full, y_train)
imp = pd.Series(pilot.feature_importances_, index=X_tr_full.columns)
selected = imp[imp >= imp.quantile(best['fs_quantile'])].index.tolist()

X_tr_f, X_te_f = X_tr_full[selected], X_te_full[selected]

# Target Transform
qt = QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=42)
y_tr_t = qt.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Final Base Models
m1 = XGBRegressor(
    n_estimators=best['xgb_n'], max_depth=best['xgb_depth'], 
    learning_rate=best['xgb_lr'], subsample=best['xgb_sub'], 
    reg_lambda=best['xgb_lambda'], random_state=42
).fit(X_tr_f, y_tr_t)

m2 = LGBMRegressor(
    n_estimators=best['lgb_n'], num_leaves=best['lgb_leaves'], 
    learning_rate=best['lgb_lr'], reg_alpha=best['lgb_alpha'], 
    random_state=42, verbose=-1
).fit(X_tr_f, y_tr_t)

m3 = CatBoostRegressor(
    iterations=best['cat_n'], depth=best['cat_depth'], 
    learning_rate=best['cat_lr'], l2_leaf_reg=best['cat_l2'], 
    random_state=42, verbose=0
).fit(X_tr_f, y_tr_t)

# Final Blended Prediction (Using Best Weights)
total_w = best['w_xgb'] + best['w_lgb'] + best['w_cat']
p1_te, p2_te, p3_te = m1.predict(X_te_f), m2.predict(X_te_f), m3.predict(X_te_f)
test_preds_t = (best['w_xgb']*p1_te + best['w_lgb']*p2_te + best['w_cat']*p3_te) / total_w
test_preds = qt.inverse_transform(test_preds_t.reshape(-1, 1)).ravel()

# Results
print("\n--- FINAL HOLDOUT PERFORMANCE ---")
print(f"R2:   {r2_score(y_test, test_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, test_preds)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, test_preds):.4f}")