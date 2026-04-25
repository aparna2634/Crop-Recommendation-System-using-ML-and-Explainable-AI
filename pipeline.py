"""
=============================================================================
  CROP RECOMMENDATION SYSTEM — END-TO-END PIPELINE  (Steps 1 – 8)
  Dataset : Crop_recommendation.csv  (Kaggle — Atharva Ingle)
            https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
  Columns : N, P, K, temperature, humidity, ph, rainfall, label
  Crops   : 22  |  Rows : 2 200

  Steps covered:
    1. Data loading & dataset features
    2. Data preprocessing (missing-value handling, encoding, scaling)
    3. Exploratory Data Analysis
    4. Train-test split
    5. Model training  (Random Forest + ANN/MLP)
    6. Model evaluation (accuracy, confusion matrix, classification report)
    7. Explainable AI  (SHAP + LIME)
    8. Model saving    (joblib)

  Run:  python pipeline.py
  Then: streamlit run app.py
=============================================================================
"""

import os
import sys
import warnings
import json

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.ensemble        import RandomForestClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, ConfusionMatrixDisplay)
from lime.lime_tabular        import LimeTabularExplainer

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH     = "Crop_recommendation.csv"   # place this file next to pipeline.py
OUTPUT_DIR   = "crop_model_artifacts"
PLOT_DIR     = os.path.join(OUTPUT_DIR, "plots")
FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL   = "label"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)


# =============================================================================
# STEP 1 — DATA LOADING & DATASET FEATURES
# =============================================================================
print("\n" + "="*70)
print("STEP 1 — DATA LOADING & DATASET FEATURES")
print("="*70)

if not os.path.exists(CSV_PATH):
    print(f"\n  ERROR: '{CSV_PATH}' not found.")
    print("  Download it from Kaggle and place it next to pipeline.py:")
    print("  https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset\n")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

# Validate columns
expected_cols = set(FEATURE_COLS + [TARGET_COL])
missing_cols  = expected_cols - set(df.columns)
if missing_cols:
    print(f"  ERROR: Missing columns in CSV: {missing_cols}")
    print(f"  Found: {list(df.columns)}")
    sys.exit(1)

print(f"\n  File loaded     : {CSV_PATH}")
print(f"  Shape           : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Crops (classes) : {df[TARGET_COL].nunique()} => {sorted(df[TARGET_COL].unique())}")
print(f"  Missing values  : {df.isnull().sum().sum()}")
print(f"\n  First 5 rows:")
print(df.head().to_string())
print(f"\n  Feature statistics:")
print(df[FEATURE_COLS].describe().round(4).to_string())
print(f"\n  Samples per crop:")
print(df[TARGET_COL].value_counts().sort_index().to_string())


# =============================================================================
# STEP 2 — DATA PREPROCESSING
# =============================================================================
print("\n" + "="*70)
print("STEP 2 — DATA PREPROCESSING")
print("="*70)

# 2-a  Duplicate removal
n_dupes = df.duplicated().sum()
print(f"\n  Duplicate rows : {n_dupes}")
if n_dupes > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  After removal  : {len(df)} rows")

# 2-b  Missing value handling — median imputation
print(f"\n  Missing values per feature (before imputation):")
print(df[FEATURE_COLS].isnull().sum().to_string())
for col in FEATURE_COLS:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col]    = df[col].fillna(median_val)
        print(f"  '{col}' imputed with median = {median_val:.4f}")
print(f"  Missing after imputation: {df[FEATURE_COLS].isnull().sum().sum()}")

# 2-c  Label encoding
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df[TARGET_COL])
print(f"\n  Label encoding complete ({len(le.classes_)} classes):")
for i, cls in enumerate(le.classes_):
    print(f"    {i:2d} -> {cls}")

# 2-d  Feature scaling (StandardScaler)
X_raw    = df[FEATURE_COLS].values.astype(float)
y        = df["label_enc"].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"\n  StandardScaler fitted:")
for col, mean, std in zip(FEATURE_COLS, scaler.mean_, scaler.scale_):
    print(f"    {col:<15} mean={mean:9.4f}  std={std:9.4f}")


# =============================================================================
# STEP 3 — EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("STEP 3 — EXPLORATORY DATA ANALYSIS")
print("="*70)

# --- 3-a  Crop distribution ---
counts = df[TARGET_COL].value_counts().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(counts.index, counts.values, color="steelblue", edgecolor="white")
for bar, val in zip(bars, counts.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=9)
ax.set_title("Crop Distribution in Dataset", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Samples")
ax.set_xlim(0, counts.max() * 1.12)
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "01_crop_distribution.png"), dpi=130)
plt.close(fig)
print("  -> Saved: 01_crop_distribution.png")

# --- 3-b  Feature histograms with KDE ---
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, col in enumerate(FEATURE_COLS):
    axes[i].hist(df[col], bins=40, color="coral",
                 edgecolor="white", density=True, alpha=0.75)
    df[col].plot.kde(ax=axes[i], color="darkred", linewidth=1.5)
    axes[i].set_title(col, fontweight="bold", fontsize=11)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Density")
    axes[i].grid(axis="y", alpha=0.3)
axes[-1].set_visible(False)
fig.suptitle("Feature Distributions (Histogram + KDE)", fontsize=15, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "02_feature_distributions.png"), dpi=130)
plt.close(fig)
print("  -> Saved: 02_feature_distributions.png")

# --- 3-c  Correlation heatmap ---
fig, ax = plt.subplots(figsize=(9, 7))
corr = df[FEATURE_COLS].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, square=True, vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "03_correlation_heatmap.png"), dpi=130)
plt.close(fig)
print("  -> Saved: 03_correlation_heatmap.png")

# --- 3-d  Box plots per feature (all 22 crops) ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
crop_order = sorted(df[TARGET_COL].unique())
for i, col in enumerate(FEATURE_COLS):
    crop_groups = [df.loc[df[TARGET_COL] == c, col].values for c in crop_order]
    bp = axes[i].boxplot(crop_groups, patch_artist=True, notch=False,
                         medianprops=dict(color="black", linewidth=1.5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(crop_groups)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[i].set_title(col, fontweight="bold", fontsize=11)
    axes[i].set_xticks(range(1, len(crop_order) + 1))
    axes[i].set_xticklabels(crop_order, rotation=75, fontsize=6.5)
    axes[i].grid(axis="y", alpha=0.3)
axes[-1].set_visible(False)
fig.suptitle("Feature Distribution by Crop (Box Plots)", fontsize=15, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "04_boxplots_by_crop.png"), dpi=130)
plt.close(fig)
print("  -> Saved: 04_boxplots_by_crop.png")

# --- 3-e  Pair plot (N, P, K, rainfall) ---
pair_cols = ["N", "P", "K", "rainfall", TARGET_COL]
pair_df   = df[pair_cols].copy()
pg = sns.pairplot(pair_df, hue=TARGET_COL,
                  plot_kws={"alpha": 0.4, "s": 14},
                  diag_kind="kde", corner=True, palette="tab20")
pg.fig.suptitle("Pair Plot — N, P, K, Rainfall  (coloured by crop)",
                y=1.01, fontsize=13, fontweight="bold")
pg.fig.savefig(os.path.join(PLOT_DIR, "05_pairplot.png"),
               dpi=110, bbox_inches="tight")
plt.close(pg.fig)
print("  -> Saved: 05_pairplot.png")

# --- 3-f  Mean feature values per crop (heatmap) ---
crop_means      = df.groupby(TARGET_COL)[FEATURE_COLS].mean()
crop_means_norm = (crop_means - crop_means.min()) / (crop_means.max() - crop_means.min())
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(crop_means_norm, annot=crop_means.round(1), fmt=".1f",
            cmap="YlGn", linewidths=0.4, ax=ax,
            cbar_kws={"label": "Normalised value (0-1)"})
ax.set_title("Mean Feature Values per Crop  (colour = normalised, annotation = actual)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Feature")
ax.set_ylabel("Crop")
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "06_crop_feature_heatmap.png"), dpi=130)
plt.close(fig)
print("  -> Saved: 06_crop_feature_heatmap.png")

print("\n  EDA complete.")


# =============================================================================
# STEP 4 — TRAIN-TEST SPLIT
# =============================================================================
print("\n" + "="*70)
print("STEP 4 — TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")
print(f"  Features         : {X_train.shape[1]}")
print(f"  Split            : 80/20 stratified")


# =============================================================================
# STEP 5 — MODEL TRAINING
# =============================================================================
print("\n" + "="*70)
print("STEP 5 — MODEL TRAINING")
print("="*70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---- Random Forest -----------------------------------------------------------
print("\n  [Random Forest]")
rf_model = RandomForestClassifier(
    n_estimators      = 200,
    max_depth         = None,
    min_samples_split = 2,
    min_samples_leaf  = 1,
    max_features      = "sqrt",
    random_state      = 42,
    n_jobs            = -1,
)
rf_model.fit(X_train, y_train)
rf_cv = cross_val_score(rf_model, X_train, y_train,
                        cv=cv, scoring="accuracy", n_jobs=-1)
print(f"  5-fold CV  : {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}")
print(f"  Fold scores: {[round(s, 4) for s in rf_cv]}")

# ---- ANN / MLP ---------------------------------------------------------------
print("\n  [Artificial Neural Network — MLP]")
ann_model = MLPClassifier(
    hidden_layer_sizes  = (256, 128, 64),
    activation          = "relu",
    solver              = "adam",
    learning_rate       = "adaptive",
    learning_rate_init  = 0.001,
    max_iter            = 1000,
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 30,
    batch_size          = 32,
    random_state        = 42,
)
ann_model.fit(X_train, y_train)
ann_cv = cross_val_score(ann_model, X_train, y_train,
                         cv=cv, scoring="accuracy", n_jobs=-1)
print(f"  5-fold CV  : {ann_cv.mean():.4f} +/- {ann_cv.std():.4f}")
print(f"  Fold scores: {[round(s, 4) for s in ann_cv]}")
print(f"  Epochs ran : {ann_model.n_iter_}")

# ANN loss curve plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ann_model.loss_curve_, color="royalblue", label="Training loss")
if ann_model.validation_scores_ is not None:
    ax2 = ax.twinx()
    ax2.plot(ann_model.validation_scores_, color="darkorange",
             linestyle="--", label="Validation accuracy")
    ax2.set_ylabel("Validation Accuracy", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss", color="royalblue")
ax.tick_params(axis="y", labelcolor="royalblue")
ax.set_title("ANN Training Loss Curve", fontsize=12, fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "07_ann_loss_curve.png"), dpi=130)
plt.close(fig)
print("  -> Saved: 07_ann_loss_curve.png")


# =============================================================================
# STEP 6 — MODEL EVALUATION
# =============================================================================
print("\n" + "="*70)
print("STEP 6 — MODEL EVALUATION")
print("="*70)

def evaluate_model(model, name, X_tr, X_te, y_tr, y_te, le, plot_dir):
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    train_acc = accuracy_score(y_tr, y_pred_tr)
    test_acc  = accuracy_score(y_te, y_pred_te)

    print(f"\n  {'─'*60}")
    print(f"  {name}")
    print(f"  {'─'*60}")
    print(f"  Train accuracy : {train_acc*100:.2f}%")
    print(f"  Test  accuracy : {test_acc*100:.2f}%")
    print(f"\n  Classification Report (test set):")
    print(classification_report(y_te, y_pred_te,
                                target_names=le.classes_, zero_division=0))

    # Confusion matrix
    cm   = confusion_matrix(y_te, y_pred_te)
    fig, ax = plt.subplots(figsize=(16, 13))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45,
              cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {name}\n(Test Accuracy: {test_acc*100:.2f}%)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    safe  = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    fname = f"cm_{safe}.png"
    fig.savefig(os.path.join(plot_dir, fname), dpi=120)
    plt.close(fig)
    print(f"  -> Confusion matrix saved: {fname}")
    return test_acc

rf_acc  = evaluate_model(rf_model,  "Random Forest",
                         X_train, X_test, y_train, y_test, le, PLOT_DIR)
ann_acc = evaluate_model(ann_model, "Artificial Neural Network MLP",
                         X_train, X_test, y_train, y_test, le, PLOT_DIR)

# Feature importance plot (RF)
fi = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(fi)))
bars = ax.barh(fi.index, fi.values, color=bar_colors, edgecolor="white")
for bar, val in zip(bars, fi.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_title("Random Forest Feature Importances (Gini)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.set_xlim(0, fi.max() * 1.18)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "08_rf_feature_importances.png"), dpi=130)
plt.close(fig)
print("\n  -> Saved: 08_rf_feature_importances.png")

# Model comparison bar
fig, ax = plt.subplots(figsize=(7, 4))
names  = ["Random Forest", "ANN (MLP)"]
accs   = [rf_acc * 100, ann_acc * 100]
clrs   = ["#2e7d32", "#1565c0"]
bars   = ax.bar(names, accs, color=clrs, edgecolor="white", width=0.45)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{acc:.2f}%", ha="center", fontweight="bold", fontsize=12)
ax.set_ylim(0, 110)
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Model Comparison — Test Accuracy", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "09_model_comparison.png"), dpi=130)
plt.close(fig)
print("  -> Saved: 09_model_comparison.png")

best_model_name = "Random Forest" if rf_acc >= ann_acc else "ANN"
best_acc        = max(rf_acc, ann_acc)
print(f"\n  Best model: {best_model_name}  (test acc = {best_acc*100:.2f}%)")


# =============================================================================
# STEP 7 — EXPLAINABLE AI: SHAP + LIME
# =============================================================================
print("\n" + "="*70)
print("STEP 7 — EXPLAINABLE AI  (SHAP + LIME)")
print("="*70)

# ---- SHAP -------------------------------------------------------------------
print("\n  [SHAP — TreeExplainer on Random Forest]")

bg_size = min(200, len(X_train))
bg_idx  = np.random.choice(len(X_train), size=bg_size, replace=False)
shap_explainer = shap.TreeExplainer(
    rf_model,
    data                 = X_train[bg_idx],
    feature_perturbation = "interventional",
)

shap_n    = min(150, len(X_test))
shap_idx  = np.random.choice(len(X_test), size=shap_n, replace=False)
X_shap    = X_test[shap_idx]
shap_vals = shap_explainer.shap_values(X_shap)

# Handle list (old shap) vs 3-D array (new shap)
if isinstance(shap_vals, list):
    mean_shap_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
else:
    mean_shap_abs = np.mean(np.abs(shap_vals), axis=-1)

mean_shap_per_feature = mean_shap_abs.mean(axis=0)
sorted_shap_idx       = np.argsort(mean_shap_per_feature)[::-1]

# Global SHAP bar chart
fig, ax = plt.subplots(figsize=(8, 5))
clrs = plt.cm.plasma(np.linspace(0.2, 0.8, len(FEATURE_COLS)))
ax.barh(
    [FEATURE_COLS[i] for i in sorted_shap_idx[::-1]],
    mean_shap_per_feature[sorted_shap_idx[::-1]],
    color=clrs[::-1], edgecolor="white"
)
ax.set_xlabel("Mean |SHAP value|  (across all classes and test samples)")
ax.set_title("SHAP Global Feature Importance\n(Random Forest, test set)",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "10_shap_global.png"), dpi=130)
plt.close(fig)
print("  -> Saved: 10_shap_global.png")

print("\n  SHAP global feature ranking:")
for rank, i in enumerate(sorted_shap_idx, 1):
    print(f"    {rank}. {FEATURE_COLS[i]:<15} mean|SHAP| = {mean_shap_per_feature[i]:.5f}")

print("""
  Interpretation guide (SHAP):
    - Positive SHAP  =>  feature pushes prediction TOWARDS a crop class.
    - Negative SHAP  =>  feature pushes prediction AWAY from a crop class.
    - Mean |SHAP|    =>  average absolute impact across all classes/samples.
    - Values are additive: they sum to the model's total output shift from
      the baseline (average prediction over the background dataset).
""")

# ---- LIME -------------------------------------------------------------------
print("  [LIME — LimeTabularExplainer on both models]")

lime_explainer = LimeTabularExplainer(
    training_data         = X_train,
    feature_names         = FEATURE_COLS,
    class_names           = list(le.classes_),
    mode                  = "classification",
    discretize_continuous = True,
    random_state          = 42,
)

def run_lime(model, model_name, sample_idx=0):
    sample         = X_test[sample_idx]
    pred_class_idx = int(model.predict([sample])[0])
    pred_class     = le.classes_[pred_class_idx]

    exp = lime_explainer.explain_instance(
        data_row     = sample,
        predict_fn   = model.predict_proba,
        num_features = len(FEATURE_COLS),
        num_samples  = 1000,
        top_labels   = 3,
    )
    top_label    = (pred_class_idx if pred_class_idx in exp.available_labels()
                    else exp.available_labels()[0])
    feat_weights = dict(exp.as_list(label=top_label))

    print(f"\n    [{model_name}]  sample #0  =>  predicted: {pred_class}")
    for feat, w in sorted(feat_weights.items(), key=lambda x: abs(x[1]), reverse=True):
        arrow = "^ supports" if w > 0 else "v opposes "
        print(f"      {arrow}  {w:+.5f}   {feat}")

    sorted_pairs = sorted(feat_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    fl  = [p[0] for p in sorted_pairs]
    wts = [p[1] for p in sorted_pairs]
    clr = ["#2e7d32" if w > 0 else "#c62828" for w in wts]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(fl[::-1], wts[::-1], color=clr[::-1], edgecolor="white", height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME weight")
    ax.set_title(f"LIME Explanation — {model_name}\nPredicted: {pred_class}",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    safe  = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    fname = f"11_lime_{safe}.png"
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=130)
    plt.close(fig)
    print(f"    -> Saved: {fname}")
    return feat_weights

run_lime(rf_model,  "Random Forest",            sample_idx=0)
run_lime(ann_model, "Artificial Neural Network", sample_idx=0)

print("""
  Interpretation guide (LIME):
    - LIME creates a local linear approximation around one sample.
    - Green bars => this condition SUPPORTS the predicted crop.
    - Red bars   => this condition CONTRADICTS the predicted crop.
    - Conditions are discretised: e.g. 'humidity > 80.5' means the
      scaled input falls in that bucket for this sample.
    - LIME is LOCAL: use SHAP for the global, average picture.
""")


# =============================================================================
# STEP 8 — MODEL SAVING
# =============================================================================
print("\n" + "="*70)
print("STEP 8 — MODEL SAVING (joblib)")
print("="*70)

def save_obj(obj, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    joblib.dump(obj, path, compress=3)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {filename:<45}  ({size_kb:>9.1f} KB)")

save_obj(rf_model,       "random_forest_model.joblib")
save_obj(ann_model,      "ann_model.joblib")
save_obj(scaler,         "scaler.joblib")
save_obj(le,             "label_encoder.joblib")
save_obj(shap_explainer, "shap_explainer.joblib")

# X_train saved as .npy — LIME explainer will be rebuilt in app.py
# (LIME contains unpickleable lambdas internally)
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, "X_train.npy")) / 1024
print(f"  Saved: {'X_train.npy':<45}  ({size_kb:>9.1f} KB)")

# Metadata JSON consumed by app.py
meta = {
    "features"          : FEATURE_COLS,
    "classes"           : list(le.classes_),
    "n_samples"         : int(len(df)),
    "n_crops"           : int(df[TARGET_COL].nunique()),
    "rf_test_accuracy"  : round(float(rf_acc),  6),
    "ann_test_accuracy" : round(float(ann_acc), 6),
    "rf_cv_mean"        : round(float(rf_cv.mean()),  6),
    "rf_cv_std"         : round(float(rf_cv.std()),   6),
    "ann_cv_mean"       : round(float(ann_cv.mean()), 6),
    "ann_cv_std"        : round(float(ann_cv.std()),  6),
    "best_model"        : best_model_name,
    "feature_importances": dict(zip(FEATURE_COLS,
                                    rf_model.feature_importances_.round(6).tolist())),
    "shap_mean_abs"     : dict(zip(FEATURE_COLS,
                                   mean_shap_per_feature.round(6).tolist())),
    "feature_stats"     : {
        col: {
            "mean": round(float(df[col].mean()), 4),
            "std" : round(float(df[col].std()),  4),
            "min" : round(float(df[col].min()),  4),
            "max" : round(float(df[col].max()),  4),
            "p25" : round(float(df[col].quantile(0.25)), 4),
            "p75" : round(float(df[col].quantile(0.75)), 4),
        }
        for col in FEATURE_COLS
    },
}
meta_path = os.path.join(OUTPUT_DIR, "model_metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
size_kb = os.path.getsize(meta_path) / 1024
print(f"  Saved: {'model_metadata.json':<45}  ({size_kb:>9.1f} KB)")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("PIPELINE COMPLETE — SUMMARY")
print("="*70)
print(f"  Dataset       : {meta['n_samples']} rows x {len(FEATURE_COLS)} features x {meta['n_crops']} crops")
print(f"  Random Forest : test={rf_acc*100:.2f}%  CV={rf_cv.mean()*100:.2f}% +/- {rf_cv.std()*100:.2f}%")
print(f"  ANN (MLP)     : test={ann_acc*100:.2f}%  CV={ann_cv.mean()*100:.2f}% +/- {ann_cv.std()*100:.2f}%")
print(f"  Best model    : {best_model_name}")
print(f"  Artifacts     : ./{OUTPUT_DIR}/")
print(f"  Plots         : ./{PLOT_DIR}/  ({len(os.listdir(PLOT_DIR))} files)")
print("="*70)
print("  Next step:  streamlit run app.py")
print("="*70 + "\n")