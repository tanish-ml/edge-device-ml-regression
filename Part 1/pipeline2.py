import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

X_raw = pd.read_csv("dataset_7.csv")
y_raw = pd.read_csv("target_7.csv")["target01"]

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

TOP_MI = ['feat_112','feat_143','feat_29','feat_240','feat_219','feat_179','feat_259']

def feature_engineering(X):
    X = X.copy()

    # interactions
    for i, f1 in enumerate(TOP_MI):
        for f2 in TOP_MI[i+1:]:
            X[f"{f1}_{f2}"] = X[f1] * X[f2]

    return X

def select_features(X, y, quantile=0.87):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns)
    return imp[imp >= imp.quantile(quantile)].index.tolist()


def get_base_models(seed=42):
    return {
        "xgb": XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            random_state=seed
        ),
        "lgb": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            random_state=seed,
            verbose=-1
        ),
        "cat": CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            verbose=0,
            random_state=seed
        )
    }
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
metrics = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold+1}")

    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    # feature engineering
    X_tr = feature_engineering(X_tr)
    X_val = feature_engineering(X_val)

    # PCA (fit on train only)
    pca = PCA(n_components=10, random_state=42)
    pca_tr = pca.fit_transform(X_tr.filter(regex="feat_"))
    pca_val = pca.transform(X_val.filter(regex="feat_"))

    for i in range(10):
        X_tr[f"pca_{i}"] = pca_tr[:, i]
        X_val[f"pca_{i}"] = pca_val[:, i]

    # feature selection
    selected = select_features(X_tr, y_tr)
    X_tr = X_tr[selected]
    X_val = X_val[selected]

    # target transform
    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, len(X_tr)),
        random_state=42
    )
    y_tr_t = qt.fit_transform(y_tr.values.reshape(-1,1)).ravel()

    # base models
    base_models = get_base_models()
    meta_X_tr = np.zeros((len(X_tr), len(base_models)))
    meta_X_val = np.zeros((len(X_val), len(base_models)))

    for i, model in enumerate(base_models.values()):
        model.fit(X_tr, y_tr_t)
        meta_X_tr[:, i] = model.predict(X_tr)
        meta_X_val[:, i] = model.predict(X_val)

    # meta learner
    meta = RidgeCV()
    meta.fit(meta_X_tr, y_tr_t)

    val_preds_t = meta.predict(meta_X_val)
    val_preds = qt.inverse_transform(val_preds_t.reshape(-1,1)).ravel()

    oof_preds[val_idx] = val_preds

    metrics.append([
        np.sqrt(mean_squared_error(y_val, val_preds)),
        r2_score(y_val, val_preds),
        mean_absolute_error(y_val, val_preds)
    ])
metrics = np.array(metrics)
print("\nCV Results:")
print(f"RMSE: {metrics[:,0].mean():.4f} ± {metrics[:,0].std():.4f}")
print(f"R2:   {metrics[:,1].mean():.4f}")
print(f"MAE:  {metrics[:,2].mean():.4f}")

# ===============================
# FINAL TRAINING ON FULL TRAIN
# ===============================

# 1. Feature engineering
X_tr_full = feature_engineering(X_train)
X_te_full = feature_engineering(X_test)

# 2. PCA (fit on TRAIN ONLY)
pca = PCA(n_components=10, random_state=42)

pca_tr = pca.fit_transform(X_tr_full.filter(regex="feat_"))
pca_te = pca.transform(X_te_full.filter(regex="feat_"))

for i in range(10):
    X_tr_full[f"pca_{i}"] = pca_tr[:, i]
    X_te_full[f"pca_{i}"] = pca_te[:, i]

# 3. Feature selection (fit on TRAIN ONLY)
selected_features = select_features(X_tr_full, y_train)

X_tr_full = X_tr_full[selected_features]
X_te_full = X_te_full[selected_features]

# 4. Target transformation (fit on TRAIN ONLY)
qt = QuantileTransformer(
    output_distribution="normal",
    n_quantiles=min(1000, len(X_tr_full)),
    random_state=42
)

y_tr_full_t = qt.fit_transform(
    y_train.values.reshape(-1, 1)
).ravel()

# 5. Train base models on FULL TRAIN
base_models = get_base_models()

meta_X_tr = np.zeros((len(X_tr_full), len(base_models)))
meta_X_te = np.zeros((len(X_te_full), len(base_models)))

for i, model in enumerate(base_models.values()):
    model.fit(X_tr_full, y_tr_full_t)
    meta_X_tr[:, i] = model.predict(X_tr_full)
    meta_X_te[:, i] = model.predict(X_te_full)

# 6. Train meta-learner
meta_model = RidgeCV()
meta_model.fit(meta_X_tr, y_tr_full_t)

# 7. Predict TEST (ONCE)
test_preds_t = meta_model.predict(meta_X_te)
test_preds = qt.inverse_transform(
    test_preds_t.reshape(-1, 1)
).ravel()
print("\n--- TRUE HOLDOUT PERFORMANCE ---")
print(f"R2:   {r2_score(y_test, test_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, test_preds)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, test_preds):.4f}")
