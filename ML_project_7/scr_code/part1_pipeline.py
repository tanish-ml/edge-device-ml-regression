import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor

# 1. SETTINGS & OPTIMIZED PARAMETERS
FILE_DATA = "scr_code/problem_7/dataset_7.csv"
FILE_TARGET = "scr_code/problem_7/target_7.csv"
FILE_EVAL = "scr_code/problem_7/EVAL_7.csv"
TOP_MI = ['feat_112','feat_143','feat_29','feat_240','feat_219','feat_259']

BEST_PARAMS = {
    'fs_quantile': 0.9576647812118337,
    'pca_components': 26,
    'iterations': 1500,
    'depth': 8,
    'learning_rate': 0.04074858721679824,
    'l2_leaf_reg': 5.510871238071969,
    'bagging_temperature': 0.7183411444287063,
    'random_strength': 1.6049765464848271,
    'border_count': 254,
    'bootstrap_type': 'Bayesian',
    'random_state': 42,
    'verbose': 0
}

# 2. HELPER FUNCTIONS
def feature_engineering(X):
    X = X.copy()
    for i, f1 in enumerate(TOP_MI):
        for f2 in TOP_MI[i+1:]:
            X[f"{f1}_{f2}"] = X[f1] * X[f2]
    return X

def get_metrics(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }

# 3. DATA LOADING
X_raw = pd.read_csv(FILE_DATA)
y_raw = pd.read_csv(FILE_TARGET)["target01"]
X_eval_raw = pd.read_csv(FILE_EVAL)

# 4. CROSS-VALIDATION PIPELINE
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_metrics_list = []
test_metrics_list = []

all_y_test, all_test_preds = [], []
fold_models = [] 

print(f"Starting 5-Fold Cross-Validation & Eval Preparation...\n")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw, y_raw)):
    X_train_fold, X_val_fold = X_raw.iloc[train_idx].copy(), X_raw.iloc[val_idx].copy()
    y_train_fold, y_val_fold = y_raw.iloc[train_idx], y_raw.iloc[val_idx]
    
    # 4a. Feature Engineering
    X_train_eng = feature_engineering(X_train_fold)
    X_val_eng = feature_engineering(X_val_fold)
    
    # 4b. Scaling & PCA
    feat_cols = [c for c in X_train_eng.columns if "feat_" in c]
    
    # Initialize and Fit Scaler
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(X_train_eng[feat_cols])
    scaled_val = scaler.transform(X_val_eng[feat_cols])
    
    pca = PCA(n_components=BEST_PARAMS['pca_components'], random_state=42)
    pca_train = pca.fit_transform(scaled_train) # Using Scaled Data
    pca_val = pca.transform(scaled_val)         # Using Scaled Data
    
    pca_cols = [f"pca_{i}" for i in range(BEST_PARAMS['pca_components'])]
    X_train_eng[pca_cols] = pca_train
    X_val_eng[pca_cols] = pca_val
    
    # 4c. Feature Selection Pilot
    pilot = CatBoostRegressor(iterations=100, depth=6, verbose=0, random_state=42)
    pilot.fit(X_train_eng, y_train_fold)
    
    imp = pd.Series(pilot.get_feature_importance(), index=X_train_eng.columns)
    thresh = imp.quantile(BEST_PARAMS['fs_quantile'])
    selected_features = imp[imp >= thresh].index.tolist()
    
    # 4d. Final Model Training
    model_params = {k: v for k, v in BEST_PARAMS.items() if k not in ['fs_quantile', 'pca_components']}
    model = CatBoostRegressor(**model_params)
    model.fit(X_train_eng[selected_features], y_train_fold)
    
    # Save for Inference (Including Scaler)
    fold_models.append({
        'model': model,
        'scaler': scaler, # Saved Scaler
        'pca': pca,
        'features': selected_features,
        'feat_cols': feat_cols 
    })
    
    # 4e. Predictions & Metrics
    f_train_preds = model.predict(X_train_eng[selected_features])
    f_test_preds = model.predict(X_val_eng[selected_features])
    
    train_m = get_metrics(y_train_fold, f_train_preds)
    test_m = get_metrics(y_val_fold, f_test_preds)
    
    train_metrics_list.append(train_m)
    test_metrics_list.append(test_m)
    all_y_test.extend(y_val_fold)
    all_test_preds.extend(f_test_preds)
    
    print(f"Fold {fold+1} | Train R2: {train_m['r2']:.4f} | Test R2: {test_m['r2']:.4f}")

# 5. AGGREGATE RESULTS
avg_test_r2 = np.mean([m['r2'] for m in test_metrics_list])
avg_test_rmse = np.mean([m['rmse'] for m in test_metrics_list])
avg_test_mae = np.mean([m['mae'] for m in test_metrics_list])

avg_train_r2 = np.mean([m['r2'] for m in train_metrics_list])
avg_train_rmse = np.mean([m['rmse'] for m in train_metrics_list])
avg_train_mae = np.mean([m['mae'] for m in train_metrics_list])

print(f"\n--- FINAL CV METRICS ---")
print(f"TRAIN -> R2: {avg_train_r2:.4f} | RMSE: {avg_train_rmse:.4f} | MAE: {avg_train_mae:.4f}")
print(f"TEST  -> R2: {avg_test_r2:.4f} | RMSE: {avg_test_rmse:.4f} | MAE: {avg_test_mae:.4f}")

# 5.5 VISUALIZATION
plt.figure(figsize=(8, 6))
sns.kdeplot(all_y_test, label="Actual (Test)", color="black", lw=2, ls="--")
sns.kdeplot(all_test_preds, label="Predicted (Test)", color="red", alpha=0.6)
plt.title("Distribution: Actual vs Predicted")
plt.legend()


plt.tight_layout()
plt.show()

# 6. EVALUATION PREDICTIONS
print(f"\nGenerating predictions for {FILE_EVAL}...")
eval_preds_accumulated = np.zeros(len(X_eval_raw))

for i, fold_pack in enumerate(fold_models):
    X_eval_eng = feature_engineering(X_eval_raw)
    
    # 1. Scale Eval features using the fold's specific scaler
    scaled_eval = fold_pack['scaler'].transform(X_eval_eng[fold_pack['feat_cols']])
    
    # 2. Apply PCA using the fold's specific PCA fit
    pca_eval = fold_pack['pca'].transform(scaled_eval)
    
    pca_cols = [f"pca_{j}" for j in range(BEST_PARAMS['pca_components'])]
    X_eval_eng[pca_cols] = pca_eval
    
    # Predict
    fold_eval_preds = fold_pack['model'].predict(X_eval_eng[fold_pack['features']])
    eval_preds_accumulated += fold_eval_preds

final_eval_preds = eval_preds_accumulated / len(fold_models)

# 7. SAVE RESULTS
pd.DataFrame({'target01': final_eval_preds}).to_csv("EVAL_target01_7.csv", index=False)
print("Results saved to EVAL_target01_7.csv")

# 8. VISUALIZATION
plt.figure(figsize=(8, 6))
sns.kdeplot(final_eval_preds, color="green", fill=True, label="EVAL Predictions")
plt.title("Distribution of Predicted Values (EVAL_7)")
plt.xlabel("Predicted target01")
plt.legend()
plt.show()  