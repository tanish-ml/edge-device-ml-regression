import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load data
X_raw = pd.read_csv('dataset_7.csv')
y_raw = pd.read_csv('target_7.csv')['target01']

# Feature Engineering
X_eng = X_raw.copy()
top_mi = ['feat_112', 'feat_143', 'feat_29', 'feat_240', 'feat_219','feat_179', 'feat_259',]
for i, f1 in enumerate(top_mi):
    # X_eng[f'log_{f1}'] = np.log1p(X_raw[f1])
    # X_eng[f'{f1}_sq'] = X_raw[f1] ** 2
    # X_eng[f'{f1}_sqrt'] = np.sqrt(np.abs(X_raw[f1]))
    # X_eng[f'{f1}_rank'] = X_raw[f1].rank(pct=True)
    for f2 in top_mi[i+1:]:
        X_eng[f'{f1}_{f2}'] = X_raw[f1] * X_raw[f2]
        # X_eng[f'{f1}_div_{f2}'] = X_raw[f1] / (X_raw[f2] + 1e-6)
        # X_eng[f'{f1}_minus_{f2}'] = X_raw[f1] - X_raw[f2]
        # X_eng[f'{f1}_absdiff_{f2}'] = (X_raw[f1] - X_raw[f2]).abs()

pca = PCA(n_components=10, random_state=42)
X_eng[[f'pca_{i}' for i in range(10)]] = pca.fit_transform(X_raw.filter(regex='feat_'))

# Robust Selection (Top 2%)
pilot = XGBRegressor(n_estimators=100, random_state=42)
pilot.fit(X_eng, y_raw)
importances = pd.Series(pilot.feature_importances_, index=X_eng.columns)
sig_features = importances[importances >= importances.quantile(0.98)].index.tolist()
X_final = X_eng[sig_features]
print(X_final.shape)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
estimator = 1000
cv_metrics = {'RMSE': [], 'R2': [], 'MAE': []}
X_train_full, X_test_holdout, y_train_full, y_test_holdout = train_test_split(X_final, y_raw, test_size=0.2, random_state=42)

# Training Loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
    X_fold_train, X_fold_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_fold_train, y_fold_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
    
    # Target Transformation (Fit only on training fold)
    qt_fold = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)
    y_fold_train_trans = qt_fold.fit_transform(y_fold_train.values.reshape(-1, 1)).flatten()
    y_fold_val_trans = qt_fold.transform(y_fold_val.values.reshape(-1, 1)).flatten()

    # Base Models with Early Stopping defined inside the loop
    # Note: For stacking, we train base models then the meta-learner
    xgb = XGBRegressor(n_estimators= estimator, learning_rate=0.02, max_depth=5, subsample=0.8, random_state=42, early_stopping_rounds=50)
    lgb = LGBMRegressor(n_estimators= estimator, learning_rate=0.02, num_leaves=31, random_state=42, verbose=-1)
    cat = CatBoostRegressor(iterations= estimator, learning_rate=0.02, depth=6, verbose=0, random_state=42, early_stopping_rounds=50)

    # Fit base models manually to use early stopping
    xgb.fit(X_fold_train, y_fold_train_trans, eval_set=[(X_fold_val, y_fold_val_trans)], verbose=False)
    cat.fit(X_fold_train, y_fold_train_trans, eval_set=[(X_fold_val, y_fold_val_trans)])
    lgb.fit(X_fold_train, y_fold_train_trans) # LGBM requires a different API for native early stopping in scikit-learn

    # Stacking logic for this fold
    base_models = [('xgb', xgb), ('lgb', lgb), ('cat', cat)]
    stack = StackingRegressor(estimators=base_models, final_estimator=RidgeCV(), cv='prefit')
    stack.fit(X_fold_train, y_fold_train_trans)
    
    # Validation
    val_preds_trans = stack.predict(X_fold_val).reshape(-1, 1)
    val_preds = qt_fold.inverse_transform(val_preds_trans).flatten()
    
    cv_metrics['RMSE'].append(np.sqrt(mean_squared_error(y_fold_val, val_preds)))
    cv_metrics['R2'].append(r2_score(y_fold_val, val_preds))
    cv_metrics['MAE'].append(mean_absolute_error(y_fold_val, val_preds))

# Final training on all 80% training data for the holdout test
qt_final = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)
y_train_full_trans = qt_final.fit_transform(y_train_full.values.reshape(-1, 1)).flatten()

final_base = [
    ('xgb', XGBRegressor(n_estimators= estimator, learning_rate=0.02, max_depth=5, subsample=0.8, random_state=42)),
    ('lgb', LGBMRegressor(n_estimators= estimator, learning_rate=0.02, num_leaves=31, random_state=42, verbose=-1)),
    ('cat', CatBoostRegressor(iterations= estimator, learning_rate=0.02, depth=6, verbose=0, random_state=42))
]
final_stack = StackingRegressor(estimators=final_base, final_estimator=RidgeCV())
final_stack.fit(X_train_full, y_train_full_trans)


print("--- Cross-Validation Stability ---")
for metric, values in cv_metrics.items():
    print(f"{metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

# Overfitting Check
train_preds_trans = final_stack.predict(X_train_full).reshape(-1, 1)
train_preds = qt_final.inverse_transform(train_preds_trans).flatten()
holdout_preds_trans = final_stack.predict(X_test_holdout).reshape(-1, 1)
holdout_preds = qt_final.inverse_transform(holdout_preds_trans).flatten()

print("\n--- Overfitting Diagnosis ---")
print(f"Train R2: {r2_score(y_train_full, train_preds):.4f}")
print(f"Test R2:  {r2_score(y_test_holdout, holdout_preds):.4f}")
