import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lineartree import LinearTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")



# 1. Prepare Data
df_features = pd.read_csv('scr_code/problem_7/dataset_7.csv')
df_targets = pd.read_csv('scr_code/problem_7/target_7.csv')
top_features = ['feat_49', 'feat_219', 'feat_111','feat_22',"feat_272", 'feat_131', 'feat_116','feat_269','feat_91','feat_185', 'feat_99', 'feat_120', 'feat_134', 'feat_196','feat_153']
X = df_features[top_features]
y = df_targets['target02']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)


# 1. Transform features to Degree 2 (results in 14 columns)
poly = PolynomialFeatures(degree=1, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 2. Initialize with a clean LinearRegression
# The model will now expect exactly 14 features based on X_train_poly
model_tree = LinearTreeRegressor(
    base_estimator=LinearRegression(),
    max_depth=6,  # Depth 3 creates 8 continuous linear regions
    min_samples_leaf=100
)
model_tree.fit(X_train_poly, y_train)
# 3.Predict


y_pred_test = model_tree.predict(X_test_poly) # Ensure you use X_test_poly here!
y_pred_train = model_tree.predict(X_train_poly)


# # 4. Visualization

# Create a figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Subplot 1: Train Data --- (Distribution Plot)
sns.kdeplot(y_train, label='Train target02', color='red', alpha=0.3, ax=axes[0])
sns.kdeplot(y_pred_train, label='Prediction Train', color='orange', linewidth=2, ax=axes[0])

axes[0].set_title('Train Distribution: Actual vs. Predictions')
axes[0].set_xlabel('target02 Value')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# --- Subplot 2: Test Data --- (Distribution Plot)
sns.kdeplot(y_test, label='Test target02', color='green', alpha=0.3, ax=axes[1])
sns.kdeplot(y_pred_test, label='Prediction Test', color='blue', linewidth=2, ax=axes[1])

axes[1].set_title('Test Distribution: Actual vs. Predictions')
axes[1].set_xlabel('target02 Value')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# 4.5 Residuals Plot
residuals = y_test - y_pred_test
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=40, kde=True)
plt.title("Test Residuals")
plt.xlabel("Residual")
plt.show()


# METRIC 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

print("TRAIN PERFORMANCE")
print("RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("R²:", r2_score(y_train, y_pred_train))

print("\nTEST PERFORMANCE")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("R²:", r2_score(y_test, y_pred_test))


#RULE EXTRACTOR

# Build a feature name list that matches the model's training dimensionality
n_features = getattr(model_tree, 'n_features_in_', None)
feature_names_full = None

# 1) Prefer names remembered by the estimator (sklearn-style)
if hasattr(model_tree, 'feature_names_in_'):
    feature_names_full = list(model_tree.feature_names_in_)

# 2) Fall back to DataFrame columns if they match the fitted dimensionality
if feature_names_full is None:
    try:
        if isinstance(X, pd.DataFrame) and (n_features is None or X.shape[1] == n_features):
            feature_names_full = list(X.columns)
    except NameError:
        pass

# 3) If polynomial features were used, try to derive expanded names
if feature_names_full is None:
    try:
        base_cols = list(X.columns) if isinstance(X, pd.DataFrame) else [f'x{i}' for i in range(n_features or 0)]
        if 'poly' in globals():
            poly_names = poly.get_feature_names_out(base_cols)
            if n_features is None or len(poly_names) == n_features:
                feature_names_full = list(poly_names)
    except Exception:
        pass

# 4) Last resort: generic names
if feature_names_full is None:
    feature_names_full = [f'feat_{i}' for i in range(n_features or 0)]

# Use full, consistent feature names (not a subset like top_features)
tree_rules = model_tree.summary(feature_names=feature_names_full)
print(tree_rules)


def get_exact_paths(model):
    # This reaches into the underlying structure you provided in the first prompt
    tree = model.summary() 
    paths = {}

    def recurse(node_id, current_path):
        node = tree[node_id]
        if 'children' in node:
            col = node['col']
            th = node['th']
            left, right = node['children']
            # Left child is <= threshold
            recurse(left, current_path + [(col, '<=', th)])
            # Right child is > threshold
            recurse(right, current_path + [(col, '>', th)])
        else:
            paths[node_id] = current_path

    recurse(0, [])
    return paths

# Get the rules
all_rules = get_exact_paths(model_tree)

# Print them out formatted
for leaf_id, criteria in all_rules.items():
    rule_str = " AND ".join([f"{c[0]} {c[1]} {c[2]:.4f}" for c in criteria])
    print(f"Leaf {leaf_id}: {rule_str}")

# Assuming model_tree.summary() is the dictionary causing the error
summary_dict = model_tree.summary()

print("--- EXACT MODEL TREE RULES ---")

for node_id, info in summary_dict.items():
    # In the linear-tree dictionary, leaves are nodes that have a 'models' object 
    # but no 'children' (or 'models' is a single estimator, not a tuple)
    if isinstance(info.get('models'), LinearRegression):
        print(f"\nLEAF NODE: {node_id}")
        
        # 1. THE RULE (The path taken to get here)
        # Check if your dict has a 'path' or 'Rule' key. 
        # If not, the path is built from the parent splits.
        rule = info.get('path', info.get('Rule', 'Path not found in dict'))
        print(f"  RULE: {rule}")
        
        # 2. THE LINEAR EQUATION
        model = info['models']
        intercept = model.intercept_
        coefs = model.coef_
        
        equation = f"  EQUATION: y = {intercept:.6f}"
        # Align coefficient names with the full feature set used for training
        for feat, val in zip(feature_names_full, coefs):
            equation += f" + ({val:.6f} * {feat})"
        
        print(equation)
        print(f"  SAMPLES IN SEGMENT: {info.get('samples', 'N/A')}")
        print("-" * 40)