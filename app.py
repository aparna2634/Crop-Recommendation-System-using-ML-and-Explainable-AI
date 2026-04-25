
import os
import json
import warnings

import numpy             as np
import pandas            as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap
import streamlit         as st
from lime.lime_tabular    import LimeTabularExplainer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "🌾 Crop Recommendation System",
    page_icon  = "🌾",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

ARTIFACT_DIR = "crop_model_artifacts"

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── global ── */
    .block-container { padding-top: 1.5rem; }

    /* ── header ── */
    .main-header {
        font-size: 2.5rem; font-weight: 900; color: #1a6b2a;
        text-align: center; letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1rem; color: #666; text-align: center; margin-bottom: 1.2rem;
    }

    /* ── prediction card ── */
    .crop-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 6px solid #2e7d32; border-radius: 14px;
        padding: 1.4rem 1.8rem; margin: 0.8rem 0;
    }
    .crop-name  { font-size: 2.1rem; font-weight: 900; color: #1b5e20;
                  text-transform: capitalize; }
    .conf-badge { display:inline-block; background:#2e7d32; color:#fff;
                  font-size:1rem; font-weight:700; padding:0.3rem 1rem;
                  border-radius:20px; margin-top:0.4rem; }
    .crop-tip   { margin-top:0.7rem; font-size:0.95rem; color:#2e7d32; }

    /* ── section titles ── */
    .section-title {
        font-size: 1.2rem; font-weight: 800; color: #2e7d32;
        border-bottom: 2px solid #a5d6a7; padding-bottom: 0.25rem;
        margin: 1.2rem 0 0.6rem 0;
    }

    /* ── metric box ── */
    .metric-box {
        background:#f1f8e9; border:1px solid #aed581;
        border-radius:8px; padding:0.7rem 1rem; margin:0.3rem 0;
        font-size:0.9rem; line-height:1.7;
    }

    /* ── primary button ── */
    .stButton > button {
        background:#2e7d32 !important; color:#fff !important;
        font-size:1rem; font-weight:700; border-radius:8px;
        padding:0.6rem 1.5rem; border:none; width:100%;
    }
    .stButton > button:hover { background:#1b5e20 !important; }

    /* ── top-5 chip ── */
    .chip {
        text-align:center; padding:0.5rem 0.3rem;
        background:#f1f8e9; border-radius:10px;
        border:1px solid #a5d6a7; margin-bottom:0.2rem;
    }
    .chip .rank  { font-size:0.7rem; color:#888; }
    .chip .emoji { font-size:1.5rem; }
    .chip .cname { font-weight:700; font-size:0.82rem; color:#1b5e20; }
    .chip .prob  { color:#2e7d32; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOOKUP TABLES
# ─────────────────────────────────────────────────────────────────────────────
CROP_EMOJI = {
    "apple":"🍎","banana":"🍌","blackgram":"🫘","chickpea":"🟡",
    "coconut":"🥥","coffee":"☕","cotton":"🌿","grapes":"🍇",
    "jute":"🌾","kidneybeans":"🫘","lentil":"🟤","maize":"🌽",
    "mango":"🥭","mothbeans":"🫘","mungbean":"🟢","muskmelon":"🍈",
    "orange":"🍊","papaya":"🍑","pigeonpeas":"🟡","pomegranate":"🍎",
    "rice":"🍚","watermelon":"🍉",
}

CROP_TIPS = {
    "apple":       "Best in cool climates (15–24 °C). Well-drained loamy soil, pH 5.5–6.5.",
    "banana":      "Tropical/sub-tropical. Needs high humidity and regular watering. pH 5.5–7.0.",
    "blackgram":   "Drought-tolerant legume. Fixes nitrogen. Well-drained loamy soil, pH 5.5–7.0.",
    "chickpea":    "Cool-season crop (10–25 °C). Well-drained soil essential. pH 6.0–9.0.",
    "coconut":     "Coastal/tropical zones. Sandy-loam with good drainage. pH 5.2–8.0.",
    "coffee":      "Shade-grown tropical highland crop. Deep well-drained acidic soil, pH 6.0–6.5.",
    "cotton":      "Requires warm climate (21–35 °C). Deep black cotton or loamy soil, pH 6.0–8.0.",
    "grapes":      "Well-drained sandy loam. Trellising improves yield. pH 5.5–7.0.",
    "jute":        "Warm humid climate. Alluvial loam soil, high rainfall zone. pH 6.0–7.5.",
    "kidneybeans": "Cool moist conditions (16–24 °C). Avoid waterlogging. pH 6.0–7.0.",
    "lentil":      "Cool semi-arid climate. Well-drained loamy soil. Good phosphorus levels key. pH 6.0–8.0.",
    "maize":       "Versatile warm-season crop. Fertile well-drained loamy soil. pH 5.8–7.0.",
    "mango":       "Tropical/sub-tropical. Deep well-drained soil. pH 5.5–7.5.",
    "mothbeans":   "Very drought-tolerant legume. Sandy or loamy soil. pH 7.0–8.5.",
    "mungbean":    "Short-duration warm crop (25–35 °C). Loamy well-drained soil. pH 6.2–7.2.",
    "muskmelon":   "Warm dry weather (25–38 °C). Sandy loam with excellent drainage. pH 6.0–7.5.",
    "orange":      "Sub-tropical (15–35 °C). Well-drained slightly acidic soil, pH 6.0–7.5.",
    "papaya":      "Year-round warm climate. Avoid waterlogging strictly. pH 6.0–7.0.",
    "pigeonpeas":  "Semi-arid tropics. Deep clay or loam soils. Drought-tolerant. pH 5.5–7.0.",
    "pomegranate": "Hot dry summers, mild winters. Well-drained slightly alkaline soil. pH 5.5–7.2.",
    "rice":        "Requires flooded paddy conditions. Clay-loam retains water well. pH 5.5–6.5.",
    "watermelon":  "Hot dry climate (25–35 °C). Sandy loam with good drainage. pH 6.0–7.5.",
}

FEATURE_LABELS = {
    "N":           "Nitrogen (N) — kg/ha",
    "P":           "Phosphorus (P) — kg/ha",
    "K":           "Potassium (K) — kg/ha",
    "temperature": "Temperature — °C",
    "humidity":    "Relative Humidity — %",
    "ph":          "Soil pH",
    "rainfall":    "Annual Rainfall — mm",
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS  (cached — runs once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models and explainers…")
def load_artifacts():
    meta_path = os.path.join(ARTIFACT_DIR, "model_metadata.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    rf   = joblib.load(os.path.join(ARTIFACT_DIR, "random_forest_model.joblib"))
    ann  = joblib.load(os.path.join(ARTIFACT_DIR, "ann_model.joblib"))
    sc   = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    le   = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.joblib"))
    shap_exp = joblib.load(os.path.join(ARTIFACT_DIR, "shap_explainer.joblib"))

    X_train  = np.load(os.path.join(ARTIFACT_DIR, "X_train.npy"))
    lime_exp = LimeTabularExplainer(
        training_data         = X_train,
        feature_names         = meta["features"],
        class_names           = meta["classes"],
        mode                  = "classification",
        discretize_continuous = True,
        random_state          = 42,
    )

    return dict(rf=rf, ann=ann, sc=sc, le=le,
                shap=shap_exp, lime=lime_exp,
                X_train=X_train, meta=meta)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def predict(art, raw_inputs: list, model_choice: str):
    arr    = np.array(raw_inputs, dtype=float).reshape(1, -1)
    scaled = art["sc"].transform(arr)
    model  = art["rf"] if model_choice == "Random Forest" else art["ann"]
    proba  = model.predict_proba(scaled)[0]
    idx    = int(np.argmax(proba))
    label  = art["le"].classes_[idx]
    conf   = float(proba[idx])
    top5_i = np.argsort(proba)[::-1][:5]
    top5   = [(art["le"].classes_[i], float(proba[i])) for i in top5_i]
    return label, conf, top5, scaled, idx


def make_shap_fig(art, scaled, pred_class_idx, features):
    sv_all = art["shap"].shap_values(scaled)
    if isinstance(sv_all, list):
        sv = sv_all[pred_class_idx][0]
    else:
        sv = sv_all[0, :, pred_class_idx]

    order = np.argsort(np.abs(sv))[::-1]
    fl    = [features[i] for i in order]
    vals  = sv[order]
    clrs  = ["#2e7d32" if v > 0 else "#c62828" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(fl[::-1], vals[::-1], color=clrs[::-1],
                   edgecolor="white", height=0.6)
    ax.axvline(0, color="#333", linewidth=0.9, linestyle="--")
    ax.set_xlabel("SHAP value  (impact on predicted crop)", fontsize=10)
    ax.set_title("SHAP Feature Contributions  (predicted class)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for bar, v in zip(bars, vals[::-1]):
        offset = 0.0005 if v >= 0 else -0.0005
        ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                f"{v:+.4f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color="#333")
    patches = [mpatches.Patch(color="#2e7d32", label="Supports prediction"),
               mpatches.Patch(color="#c62828", label="Opposes  prediction")]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    plt.tight_layout()
    return fig


def make_lime_fig(art, scaled, pred_class_idx, features, model_choice):
    model = art["rf"] if model_choice == "Random Forest" else art["ann"]
    exp   = art["lime"].explain_instance(
        data_row     = scaled[0],
        predict_fn   = model.predict_proba,
        num_features = len(features),
        num_samples  = 800,
        top_labels   = 3,
    )
    top_label = (pred_class_idx if pred_class_idx in exp.available_labels()
                 else exp.available_labels()[0])
    fw   = dict(exp.as_list(label=top_label))
    sp   = sorted(fw.items(), key=lambda x: abs(x[1]), reverse=True)
    fl   = [p[0] for p in sp]
    wts  = [p[1] for p in sp]
    clrs = ["#2e7d32" if w > 0 else "#c62828" for w in wts]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(fl[::-1], wts[::-1], color=clrs[::-1],
                   edgecolor="white", height=0.6)
    ax.axvline(0, color="#333", linewidth=0.9, linestyle="--")
    ax.set_xlabel("LIME weight", fontsize=10)
    ax.set_title("LIME Local Explanation  (this specific input)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for bar, v in zip(bars, wts[::-1]):
        offset = 0.0003 if v >= 0 else -0.0003
        ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                f"{v:+.4f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color="#333")
    patches = [mpatches.Patch(color="#2e7d32", label="Supports prediction"),
               mpatches.Patch(color="#c62828", label="Opposes  prediction")]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    plt.tight_layout()
    return fig


def make_top5_fig(top5):
    crops  = [t[0].capitalize() for t in top5]
    scores = [t[1] * 100 for t in top5]
    clrs   = ["#2e7d32"] + ["#81c784"] * (len(crops) - 1)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.barh(crops[::-1], scores[::-1], color=clrs[::-1],
                   edgecolor="white", height=0.55)
    ax.set_xlabel("Probability (%)", fontsize=10)
    ax.set_title("Top-5 Crop Recommendations", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.25)
    for bar, sc in zip(bars, scores[::-1]):
        ax.text(sc + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{sc:.1f}%", va="center", fontsize=9, color="#333")
    plt.tight_layout()
    return fig


def make_conf_gauge(conf):
    fig, ax = plt.subplots(figsize=(6, 0.65))
    ax.barh([""], [conf * 100], color="#2e7d32", height=0.45, edgecolor="white")
    ax.barh([""], [100 - conf * 100], left=[conf * 100],
            color="#e0e0e0", height=0.45, edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%","25%","50%","75%","100%"], fontsize=8)
    ax.text(conf * 50, 0, f"{conf*100:.1f}%",
            ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")
    ax.set_yticks([])
    ax.set_title("Prediction Confidence", fontsize=9)
    ax.spines[["top","right","left"]].set_visible(False)
    plt.tight_layout()
    return fig


def make_global_fi_fig(meta):
    fi      = meta["feature_importances"]
    shap_fi = meta["shap_mean_abs"]
    feats   = meta["features"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # RF Gini
    vals = [fi[f] for f in feats]
    sp   = sorted(zip(feats, vals), key=lambda x: x[1])
    ax1.barh([p[0] for p in sp], [p[1] for p in sp], color="#388e3c", edgecolor="white")
    ax1.set_title("Random Forest — Gini Importance", fontweight="bold", fontsize=10)
    ax1.set_xlabel("Importance score")
    ax1.grid(axis="x", alpha=0.25)

    # SHAP mean |value|
    sv  = [shap_fi[f] for f in feats]
    sp2 = sorted(zip(feats, sv), key=lambda x: x[1])
    ax2.barh([p[0] for p in sp2], [p[1] for p in sp2], color="#7b1fa2", edgecolor="white")
    ax2.set_title("SHAP — Global Mean |Value|", fontweight="bold", fontsize=10)
    ax2.set_xlabel("Mean |SHAP value|")
    ax2.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    return fig


def make_input_radar(raw_vals, meta, features):
    fs     = meta["feature_stats"]
    norms  = []
    for col, val in zip(features, raw_vals):
        lo = fs[col]["min"]
        hi = fs[col]["max"]
        norms.append(np.clip((val - lo) / max(hi - lo, 1e-6), 0, 1))

    clrs = ["#2e7d32" if 0.05 <= n <= 0.95 else "#e53935" for n in norms]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(features))
    bars = ax.bar(x, norms, color=clrs, edgecolor="white", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=30, fontsize=9)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["Min", "25%", "Median", "75%", "Max"], fontsize=8)
    ax.axhline(0.05, color="orange", linestyle=":", linewidth=1)
    ax.axhline(0.95, color="orange", linestyle=":", linewidth=1)
    ax.set_title("Your Inputs vs Dataset Range\n(orange lines = extreme thresholds)",
                 fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN APP
# =============================================================================
def main():

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown('<div class="main-header">🌾 Crop Recommendation System</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'AI-powered crop advisory using soil & climate data &nbsp;·&nbsp; '
        'Random Forest + ANN &nbsp;·&nbsp; SHAP + LIME explainability'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Load artifacts ────────────────────────────────────────────────────────
    artifacts = load_artifacts()
    if artifacts is None:
        st.error("❌ Model artifacts not found. Please run `pipeline.py` first.")
        st.code("python pipeline.py", language="bash")
        return

    meta     = artifacts["meta"]
    features = meta["features"]

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Model Selection")
        model_choice = st.radio(
            "Choose prediction model",
            ["Random Forest", "Artificial Neural Network"],
            index=0,
        )
        acc_key = "rf_test_accuracy"  if model_choice == "Random Forest" else "ann_test_accuracy"
        cv_key  = "rf_cv_mean"        if model_choice == "Random Forest" else "ann_cv_mean"
        cv_std  = "rf_cv_std"         if model_choice == "Random Forest" else "ann_cv_std"

        st.markdown(f"""
        <div class="metric-box">
            <b>Test accuracy</b> : {meta[acc_key]*100:.2f}%<br>
            <b>5-fold CV</b>     : {meta[cv_key]*100:.2f}% ± {meta[cv_std]*100:.2f}%<br>
            <b>Training size</b> : {meta['n_samples']} samples<br>
            <b>Best overall</b>  : {meta['best_model']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 🌱 Soil & Climate Inputs")

        fs = meta["feature_stats"]

        N = st.slider(
            "Nitrogen (N) — kg/ha",
            min_value=0, max_value=140,
            value=int(fs["N"]["mean"]),
            help=f"Dataset range: {fs['N']['min']:.0f} – {fs['N']['max']:.0f}"
        )
        P = st.slider(
            "Phosphorus (P) — kg/ha",
            min_value=0, max_value=145,
            value=int(fs["P"]["mean"]),
            help=f"Dataset range: {fs['P']['min']:.0f} – {fs['P']['max']:.0f}"
        )
        K = st.slider(
            "Potassium (K) — kg/ha",
            min_value=0, max_value=210,
            value=int(fs["K"]["mean"]),
            help=f"Dataset range: {fs['K']['min']:.0f} – {fs['K']['max']:.0f}"
        )
        temperature = st.slider(
            "Temperature — °C",
            min_value=5.0, max_value=45.0,
            value=round(fs["temperature"]["mean"], 1), step=0.1,
            help=f"Dataset range: {fs['temperature']['min']:.1f} – {fs['temperature']['max']:.1f}"
        )
        humidity = st.slider(
            "Relative Humidity — %",
            min_value=10.0, max_value=100.0,
            value=round(fs["humidity"]["mean"], 1), step=0.1,
            help=f"Dataset range: {fs['humidity']['min']:.1f} – {fs['humidity']['max']:.1f}"
        )
        ph = st.slider(
            "Soil pH",
            min_value=3.5, max_value=9.5,
            value=round(fs["ph"]["mean"], 1), step=0.1,
            help=f"Dataset range: {fs['ph']['min']:.2f} – {fs['ph']['max']:.2f}"
        )
        rainfall = st.slider(
            "Annual Rainfall — mm",
            min_value=0.0, max_value=320.0,
            value=round(fs["rainfall"]["mean"], 1), step=0.5,
            help=f"Dataset range: {fs['rainfall']['min']:.0f} – {fs['rainfall']['max']:.0f}"
        )

        st.markdown("---")
        st.markdown("## 🔧 Display Options")
        show_top5    = st.checkbox("Top-5 candidates",         value=True)
        show_shap    = st.checkbox("SHAP explanation",         value=True)
        show_lime    = st.checkbox("LIME explanation",         value=True)
        show_global  = st.checkbox("Global feature importance",value=False)
        show_eda     = st.checkbox("EDA plots from pipeline",  value=False)

        predict_btn = st.button("🔍  Predict Best Crop")

    # ── Two-column layout ─────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 2])

    raw_inputs = [N, P, K, temperature, humidity, ph, rainfall]

    with col_left:
        st.markdown('<div class="section-title">📋 Input Summary</div>',
                    unsafe_allow_html=True)
        tbl_data = {
            "Feature": list(FEATURE_LABELS.values()),
            "Your Value": [
                f"{N} kg/ha", f"{P} kg/ha", f"{K} kg/ha",
                f"{temperature} °C", f"{humidity}%",
                f"{ph}", f"{rainfall} mm"
            ],
            "Dataset Mean": [
                f"{fs['N']['mean']:.1f}", f"{fs['P']['mean']:.1f}",
                f"{fs['K']['mean']:.1f}", f"{fs['temperature']['mean']:.1f}",
                f"{fs['humidity']['mean']:.1f}", f"{fs['ph']['mean']:.2f}",
                f"{fs['rainfall']['mean']:.1f}",
            ],
        }
        st.dataframe(pd.DataFrame(tbl_data).set_index("Feature"),
                     use_container_width=True)

        st.markdown('<div class="section-title">📊 Input vs Dataset Range</div>',
                    unsafe_allow_html=True)
        fig_range = make_input_radar(raw_inputs, meta, features)
        st.pyplot(fig_range, use_container_width=True)
        plt.close(fig_range)

    with col_right:
        if not predict_btn:
            st.markdown("""
            <div style="text-align:center; padding:4rem 1rem; color:#999;">
                <div style="font-size:3.5rem;">🌱</div>
                <div style="font-size:1.1rem; margin-top:0.6rem;">
                    Adjust soil &amp; climate values in the sidebar,<br>
                    then click <b>Predict Best Crop</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # ── STEP 10: PREDICTION ────────────────────────────────────────
            with st.spinner("Running inference…"):
                crop, conf, top5, scaled, pred_idx = predict(
                    artifacts, raw_inputs, model_choice
                )

            emoji = CROP_EMOJI.get(crop, "🌿")
            tip   = CROP_TIPS.get(crop, "")

            # Prediction card
            st.markdown(f"""
            <div class="crop-card">
                <div style="font-size:0.9rem;color:#555;margin-bottom:0.2rem;">
                    ✅ &nbsp; Recommended Crop &nbsp;·&nbsp; {model_choice}
                </div>
                <div class="crop-name">{emoji}&nbsp; {crop.capitalize()}</div>
                <div class="conf-badge">Confidence: {conf*100:.1f}%</div>
                <div class="crop-tip">💡 {tip}</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence gauge
            fig_gauge = make_conf_gauge(conf)
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close(fig_gauge)

            # ── Top-5 ──────────────────────────────────────────────────────
            if show_top5:
                st.markdown('<div class="section-title">🏆 Top-5 Candidate Crops</div>',
                            unsafe_allow_html=True)
                fig_t5 = make_top5_fig(top5)
                st.pyplot(fig_t5, use_container_width=True)
                plt.close(fig_t5)

                cols_t5 = st.columns(5)
                for i, (c, p) in enumerate(top5):
                    with cols_t5[i]:
                        st.markdown(f"""
                        <div class="chip">
                            <div class="rank">#{i+1}</div>
                            <div class="emoji">{CROP_EMOJI.get(c,'🌿')}</div>
                            <div class="cname">{c.capitalize()}</div>
                            <div class="prob">{p*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

            # ── SHAP ───────────────────────────────────────────────────────
            if show_shap:
                st.markdown('<div class="section-title">🔬 SHAP Feature Contributions</div>',
                            unsafe_allow_html=True)
                with st.spinner("Computing SHAP values…"):
                    fig_shap = make_shap_fig(artifacts, scaled, pred_idx, features)
                st.pyplot(fig_shap, use_container_width=True)
                plt.close(fig_shap)

                with st.expander("ℹ️ How to interpret SHAP"):
                    st.markdown("""
**SHAP (SHapley Additive exPlanations)** uses game theory to assign each feature a
fair credit for the model's prediction.

| Colour | Meaning |
|--------|---------|
| 🟢 Green | Feature value **pushes** the model **towards** the predicted crop |
| 🔴 Red   | Feature value **pushes** the model **away** from it |

- Bar length = strength of influence for *this prediction*.
- Values are **additive**: they sum to the model's output shift from a baseline.
- Example: `humidity = 0.025` means that your humidity reading raised the
  model's log-odds for this crop by 0.025 units relative to the average.
                    """)

            # ── LIME ───────────────────────────────────────────────────────
            if show_lime:
                st.markdown('<div class="section-title">🧪 LIME Local Explanation</div>',
                            unsafe_allow_html=True)
                with st.spinner("Computing LIME explanation…"):
                    fig_lime = make_lime_fig(artifacts, scaled, pred_idx, features, model_choice)
                st.pyplot(fig_lime, use_container_width=True)
                plt.close(fig_lime)

                with st.expander("ℹ️ How to interpret LIME"):
                    st.markdown("""
**LIME (Local Interpretable Model-agnostic Explanations)** perturbs your specific
input many times and fits a *local linear model* to approximate the complex model
in that neighbourhood.

| Colour | Meaning |
|--------|---------|
| 🟢 Green | Condition **supports** predicting this crop for your inputs |
| 🔴 Red   | Condition **contradicts** this crop (overridden by other features) |

Conditions are **discretised**: e.g. `K <= -0.56` means your potassium, after
scaling, falls in the lowest bucket — indicating a low-K crop preference is likely.

**SHAP vs LIME**: SHAP is globally calibrated; LIME is purely local. Use both
together for the most complete picture.
                    """)

    # ── Global feature importance (full-width) ─────────────────────────────
    if predict_btn and show_global:
        st.markdown("---")
        st.markdown('<div class="section-title">🌍 Global Feature Importance</div>',
                    unsafe_allow_html=True)
        fig_gfi = make_global_fi_fig(meta)
        st.pyplot(fig_gfi, use_container_width=True)
        plt.close(fig_gfi)

    # ── EDA gallery ────────────────────────────────────────────────────────
    if predict_btn and show_eda:
        st.markdown("---")
        st.markdown('<div class="section-title">📊 EDA & Training Plots</div>',
                    unsafe_allow_html=True)

        plot_dir = os.path.join(ARTIFACT_DIR, "plots")
        eda_map  = {
            "Crop Distribution"       : "01_crop_distribution.png",
            "Feature Distributions"   : "02_feature_distributions.png",
            "Correlation Heatmap"     : "03_correlation_heatmap.png",
            "Box Plots by Crop"       : "04_boxplots_by_crop.png",
            "Pair Plot"               : "05_pairplot.png",
            "Crop-Feature Heatmap"    : "06_crop_feature_heatmap.png",
            "ANN Loss Curve"          : "07_ann_loss_curve.png",
            "RF Feature Importances"  : "08_rf_feature_importances.png",
            "Model Comparison"        : "09_model_comparison.png",
            "SHAP Global"             : "10_shap_global.png",
            "LIME — Random Forest"    : "11_lime_random_forest.png",
            "LIME — ANN"              : "11_lime_artificial_neural_network.png",
        }

        tab_names = list(eda_map.keys())
        tabs      = st.tabs(tab_names)
        for tab, (title, fname) in zip(tabs, eda_map.items()):
            with tab:
                path = os.path.join(plot_dir, fname)
                if os.path.exists(path):
                    st.image(path, caption=title, use_container_width=True)
                else:
                    st.info(f"'{fname}' not found — run pipeline.py to generate it.")

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; font-size:0.8rem; color:#aaa;">
        🌾 Crop Recommendation System &nbsp;·&nbsp;
        Scikit-learn · SHAP · LIME · Streamlit &nbsp;·&nbsp;
        Dataset: <a href="https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset"
                   target="_blank" style="color:#aaa;">Kaggle — Atharva Ingle</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
