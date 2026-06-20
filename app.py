import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score
)


# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BankruptcyIQ · Startup Risk Engine",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────────────────
# CUSTOM CSS — dark fintech aesthetic
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:        #080d14;
    --surface:   #0f1923;
    --border:    #1e2d3d;
    --amber:     #f0a500;
    --red:       #e53e3e;
    --green:     #38a169;
    --text:      #d4dde8;
    --muted:     #5a7184;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
    --font-body: 'Inter', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.stSelectbox > div > div,
.stNumberInput > div > div,
.stSlider > div { color: var(--text) !important; }

input, select, textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--muted) !important;
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom: 2px solid var(--amber) !important;
    background: transparent !important;
}

.stButton > button {
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    background: var(--amber) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stProgress > div > div { background: var(--amber) !important; }
hr { border-color: var(--border) !important; }
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px; }

.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--amber);
}
.kpi-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}
.kpi-value {
    font-family: var(--font-head);
    font-size: 2rem;
    font-weight: 800;
    color: var(--amber);
    line-height: 1;
}
.kpi-sub {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 6px;
}

.section-title {
    font-family: var(--font-head);
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: var(--text);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

.risk-high {
    background: rgba(229,62,62,0.12);
    border: 1px solid rgba(229,62,62,0.4);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.risk-low {
    background: rgba(56,161,105,0.12);
    border: 1px solid rgba(56,161,105,0.4);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.risk-title { font-family: var(--font-head); font-size: 1.6rem; font-weight: 800; }

.factor-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    margin: 3px;
}
.pill-risk { background: rgba(229,62,62,0.15); color: #fc8181; border: 1px solid rgba(229,62,62,0.3); }
.pill-safe { background: rgba(56,161,105,0.15); color: #68d391; border: 1px solid rgba(56,161,105,0.3); }

.hero { padding: 32px 0 24px; border-bottom: 1px solid var(--border); margin-bottom: 28px; }
.hero-tag { font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--amber); margin-bottom: 10px; }
.hero-title { font-family: var(--font-head); font-size: 2.4rem; font-weight: 800; line-height: 1.1; color: #eef2f7; }
.hero-sub { font-size: 0.95rem; color: var(--muted); margin-top: 10px; max-width: 560px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1923",
    "axes.facecolor":    "#0f1923",
    "axes.edgecolor":    "#1e2d3d",
    "axes.labelcolor":   "#d4dde8",
    "xtick.color":       "#5a7184",
    "ytick.color":       "#5a7184",
    "text.color":        "#d4dde8",
    "grid.color":        "#1e2d3d",
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.titlecolor":   "#d4dde8",
    "figure.autolayout": True,
})
AMBER = "#f0a500"
RED   = "#e53e3e"
GREEN = "#38a169"
BLUE  = "#4299e1"


# ─────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("startup.csv")
    df = df[df["Bankrupt"].notna()]
    df["Bankrupt"] = df["Bankrupt"].astype(int)
    if "Startup_ID" in df.columns:
        df.drop(columns=["Startup_ID"], inplace=True)

    df["Revenue_to_Burn"] = df["Revenue_MUSD"]         / (df["Burn_Rate_MUSD"]        + 1)
    df["Debt_to_Funding"] = df["Debt_Level_MUSD"]      / (df["Funding_Amount_MUSD"]   + 1)
    df["Cost_to_Revenue"] = df["Operational_Cost_MUSD"] / (df["Revenue_MUSD"]         + 1)
    df["Funding_per_Emp"] = df["Funding_Amount_MUSD"]   / (df["Employees"]            + 1)
    df["Revenue_per_Emp"] = df["Revenue_MUSD"]          / (df["Employees"]            + 1)

    return df

df = load_data()
X  = df.drop("Bankrupt", axis=1)
y  = df["Bankrupt"]


# ─────────────────────────────────────────────────────────
# FEATURE GROUPS
# ─────────────────────────────────────────────────────────
numerical_features = [
    "Funding_Amount_MUSD", "Revenue_MUSD", "Profit_Margin",
    "Burn_Rate_MUSD", "Employees", "Years_Active",
    "Customer_Growth_Rate", "Debt_Level_MUSD", "Operational_Cost_MUSD",
    "Revenue_to_Burn", "Debt_to_Funding", "Cost_to_Revenue",
    "Funding_per_Emp", "Revenue_per_Emp"
]
categorical_features = ["Market_Competition", "Founder_Experience", "Industry", "Region"]


# ─────────────────────────────────────────────────────────
# PREPROCESSOR
# ─────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ]), numerical_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10))
    ]), categorical_features)
])


# ─────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500, max_depth=14, min_samples_split=10,
        min_samples_leaf=5, max_features="log2",
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y.value_counts()[0] / y.value_counts()[1]),
        eval_metric="logloss", random_state=42, n_jobs=-1
    ),    
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced",random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, random_state=42),
    "SVM":                 SVC(probability=True, class_weight="balanced",random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=7)
}


# ─────────────────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────────────────
@st.cache_resource
def train_model(model_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", MODEL_REGISTRY[model_name])
    ])
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────
# BENCHMARK ALL MODELS
# ─────────────────────────────────────────────────────────
@st.cache_data
def benchmark_all_models():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    results = []
    for name, clf in MODEL_REGISTRY.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results.append({
            "Model":     name,
            "Accuracy":  accuracy_score(y_test, y_pred),
            "ROC-AUC":   auc(fpr, tpr),
            "Avg Prec":  average_precision_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred, zero_division=0),
            "F1 Score":  f1_score(y_test, y_pred, zero_division=0),
        })
    return pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Engine Settings")
model_choice = st.sidebar.selectbox("Active Model", list(MODEL_REGISTRY.keys()), index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#5a7184;line-height:1.8'>
<b style='color:#f0a500'>FEATURES</b><br>
· 13 startup input variables<br>
· 5 engineered financial ratios<br>
· Feature importance analysis<br>
· ROC / PR curves<br>
· Model benchmarking<br>
· Batch CSV scoring<br>
</div>
""", unsafe_allow_html=True)

pipeline, X_train, X_test, y_train, y_test = train_model(model_choice)


# ─────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <div class='hero-tag'>🔬 ML Risk Intelligence Platform</div>
  <div class='hero-title'>BankruptcyIQ</div>
  <div class='hero-sub'>
    Predict startup financial distress using ensemble machine learning —
    with feature importance analysis, model benchmarking, and batch scoring.
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
fpr_, tpr_, _ = roc_curve(y_test, y_prob)
roc_auc  = auc(fpr_, tpr_)
avg_prec = average_precision_score(y_test, y_prob)
bankrupt_rate = y.mean()

c1, c2, c3, c4, c5 = st.columns(5)
for col, label, value, sub in [
    (c1, "Accuracy",      f"{accuracy:.1%}",    "hold-out test set"),
    (c2, "ROC-AUC",       f"{roc_auc:.3f}",     "area under curve"),
    (c3, "Avg Precision", f"{avg_prec:.3f}",    "precision-recall"),
    (c4, "Dataset Size",  f"{len(df):,}",        "startup records"),
    (c5, "Bankrupt Rate", f"{bankrupt_rate:.1%}","class imbalance"),
]:
    col.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value'>{value}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎯  Predict",
    "📊  Evaluation",
    "⚖️  Benchmark"
])


# ══════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-title'>Startup Financial Profile</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**💰 Financials**")
        funding = st.number_input("Funding (M USD)",          0.0, 1000.0, 50.0)
        revenue = st.number_input("Revenue (M USD)",          0.0,  500.0, 20.0)
        profit  = st.slider      ("Profit Margin",           -0.5,    0.6,  0.1)
    with col2:
        st.markdown("**🔥 Operations**")
        burn    = st.number_input("Burn Rate (M USD)",        0.0,  100.0,  5.0)
        emp     = st.number_input("Employees",                   1,  10000,   50)
        years   = st.number_input("Years Active",                1,     50,    5)
    with col3:
        st.markdown("**📈 Growth & Debt**")
        growth  = st.slider      ("Customer Growth Rate",    -0.3,    0.5,  0.1)
        debt    = st.number_input("Debt (M USD)",             0.0,  500.0, 10.0)
        ops     = st.number_input("Operational Cost (M USD)", 0.0,  500.0, 20.0)

    col4, col5, col6, col7 = st.columns(4)
    market   = col4.selectbox("Market Competition", ["Low", "Medium", "High"])
    founder  = col5.selectbox("Founder Experience", ["Low", "Medium", "High"])
    industry = col6.selectbox("Industry",           ["Tech","FinTech","Health","E-commerce","AI"])
    region   = col7.selectbox("Region",             ["North America","Europe","Asia","Middle East"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("▶  Run Risk Analysis", use_container_width=True)

    if predict_clicked:
        input_df = pd.DataFrame([{
            "Funding_Amount_MUSD":   funding,
            "Revenue_MUSD":          revenue,
            "Profit_Margin":         profit,
            "Burn_Rate_MUSD":        burn,
            "Employees":             emp,
            "Years_Active":          years,
            "Customer_Growth_Rate":  growth,
            "Debt_Level_MUSD":       debt,
            "Operational_Cost_MUSD": ops,
            "Market_Competition":    market,
            "Founder_Experience":    founder,
            "Industry":              industry,
            "Region":                region
        }])
        input_df["Revenue_to_Burn"] = revenue / (burn    + 1)
        input_df["Debt_to_Funding"] = debt    / (funding + 1)
        input_df["Cost_to_Revenue"] = ops     / (revenue + 1)
        input_df["Funding_per_Emp"] = funding / (emp     + 1)
        input_df["Revenue_per_Emp"] = revenue / (emp     + 1)

        prob = pipeline.predict_proba(input_df)[0][1]
        pred = pipeline.predict(input_df)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1, 1.4, 1])

        # ── Gauge chart
        with r1:
            fig, ax = plt.subplots(figsize=(3.5, 2.2), subplot_kw={"aspect": "equal"})
            fig.patch.set_facecolor("#0f1923")
            ax.set_facecolor("#0f1923")
            for lo, hi, color in [(0.0, 0.4, GREEN), (0.4, 0.7, AMBER), (0.7, 1.0, RED)]:
                t1 = np.pi + (0 - np.pi) * lo
                t2 = np.pi + (0 - np.pi) * hi
                ts = np.linspace(t1, t2, 80)
                ax.fill_between(np.cos(ts), 0, np.sin(ts), alpha=0.25, color=color)
                ax.plot(np.cos(ts), np.sin(ts), color=color, lw=6, solid_capstyle="round")
            theta = np.pi + (0 - np.pi) * prob
            ax.annotate("", xy=(0.68 * np.cos(theta), 0.68 * np.sin(theta)),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5, mutation_scale=16))
            ax.add_patch(plt.Circle((0, 0), 0.06, color="#0f1923", zorder=5))
            ax.add_patch(plt.Circle((0, 0), 0.05, color="white",   zorder=6))
            needle_color = RED if prob > 0.6 else (AMBER if prob > 0.35 else GREEN)
            ax.text(0, -0.22, f"{prob:.1%}", ha="center", va="center",
                    fontsize=16, fontweight="bold", color=needle_color, fontfamily="monospace")
            ax.text(0, -0.42, "RISK SCORE", ha="center", fontsize=7,
                    color="#5a7184", fontfamily="monospace")
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-0.55, 1.1)
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # ── Verdict + ratio flags
        with r2:
            if pred == 1:
                level = "CRITICAL" if prob > 0.75 else "HIGH"
                st.markdown(f"""
                <div class='risk-high'>
                  <div class='risk-title' style='color:#fc8181'>⚠️ {level} RISK</div>
                  <div style='font-family:IBM Plex Mono;font-size:0.8rem;color:#5a7184;margin-top:8px'>
                    Probability of bankruptcy: <b style='color:#fc8181'>{prob:.1%}</b>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                level = "LOW" if prob < 0.25 else "MODERATE"
                st.markdown(f"""
                <div class='risk-low'>
                  <div class='risk-title' style='color:#68d391'>✅ {level} RISK</div>
                  <div style='font-family:IBM Plex Mono;font-size:0.8rem;color:#5a7184;margin-top:8px'>
                    Probability of bankruptcy: <b style='color:#68d391'>{prob:.1%}</b>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            flags = [
                ("🔴 Low Revenue/Burn",      "pill-risk") if revenue/(burn+1)  < 2    else ("🟢 Healthy Rev/Burn",       "pill-safe"),
                ("🔴 High Debt/Funding",     "pill-risk") if debt/(funding+1)  > 0.5  else ("🟢 Manageable Debt",        "pill-safe"),
                ("🔴 Costs > Revenue",       "pill-risk") if ops/(revenue+1)   > 1.2  else ("🟢 Cost Efficient",         "pill-safe"),
                ("🔴 Negative Margin",       "pill-risk") if profit < 0               else ("🟢 Positive Margin",        "pill-safe"),
                ("🔴 Stagnant Growth",       "pill-risk") if growth < 0.05            else ("🟢 Growing Customer Base",  "pill-safe"),
            ]
            pills = " ".join([f"<span class='factor-pill {cls}'>{txt}</span>" for txt, cls in flags])
            st.markdown(f"<div>{pills}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — EVALUATION
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Model Evaluation</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr",
                    xticklabels=["Healthy", "Bankrupt"],
                    yticklabels=["Healthy", "Bankrupt"],
                    linewidths=1, linecolor="#1e2d3d",
                    annot_kws={"size": 14, "weight": "bold"},
                    cbar_kws={"shrink": 0.75})
        ax.set_xlabel("Predicted", labelpad=10)
        ax.set_ylabel("Actual",    labelpad=10)
        ax.set_title(f"{model_choice} — Test Set", pad=12)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("**Classification Report**")
        report    = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**ROC Curve**")
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        ax.plot(fpr_, tpr_, color=AMBER, lw=2.5, label=f"AUC = {roc_auc:.3f}")
        ax.fill_between(fpr_, tpr_, alpha=0.08, color=AMBER)
        ax.plot([0, 1], [0, 1], "--", color="#5a7184", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:
        st.markdown("**Precision-Recall Curve**")
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        ax.plot(rec, prec, color=BLUE, lw=2.5, label=f"AP = {ap:.3f}")
        ax.fill_between(rec, prec, alpha=0.08, color=BLUE)
        ax.axhline(y=bankrupt_rate, color="#5a7184", lw=1, linestyle="--",
                   label=f"Baseline = {bankrupt_rate:.2f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Feature Importance**")
    model_step = pipeline.named_steps["model"]

    # Get feature names from the fitted preprocessor
    cat_names = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_features)
        .tolist()
    )
    all_names = numerical_features + cat_names

    if hasattr(model_step, "feature_importances_"):
        # Tree-based models (Random Forest, Gradient Boosting, Decision Tree)
        importances = model_step.feature_importances_
        n = min(len(importances), len(all_names))
        imp_df = (
            pd.DataFrame({"Feature": all_names[:n], "Score": importances[:n]})
            .sort_values("Score", ascending=False)
            .head(15)
        )
        xlabel = "Gini Importance"

    elif hasattr(model_step, "coef_"):
        # Logistic Regression
        importances = np.abs(model_step.coef_[0])
        n = min(len(importances), len(all_names))
        imp_df = (
            pd.DataFrame({"Feature": all_names[:n], "Score": importances[:n]})
            .sort_values("Score", ascending=False)
            .head(15)
        )
        xlabel = "|Coefficient|"

    else:
        imp_df = None
        xlabel = ""

    if imp_df is not None:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        colors = [AMBER if v >= imp_df["Score"].median() else "#4a6fa5"
                  for v in imp_df["Score"]]
        ax.barh(imp_df["Feature"][::-1], imp_df["Score"][::-1],
                color=colors[::-1], edgecolor="none", height=0.65)
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_title("Top 15 Feature Importances", pad=10)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right", "left"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.info("Switch to a tree-based model or Logistic Regression "
                "to see feature importances.")


# ══════════════════════════════════════════════════════════
# TAB 3 — BENCHMARK
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Model Benchmarking</div>", unsafe_allow_html=True)
    with st.spinner("Training & evaluating all models…"):
        bench_df = benchmark_all_models()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Performance Comparison**")
        st.dataframe(
            bench_df.style
                .highlight_max(
                    subset=["Accuracy", "ROC-AUC", "Avg Prec", "Precision", "Recall", "F1 Score"],
                    color="#1a2e1a"
                )
                .format({
                    "Accuracy": "{:.3f}", "ROC-AUC": "{:.3f}", "Avg Prec": "{:.3f}",
                    "Precision": "{:.3f}", "Recall": "{:.3f}", "F1 Score": "{:.3f}"
                }),
            use_container_width=True
        )

    with col2:
        st.markdown("**ROC-AUC vs Accuracy**")
        fig, ax = plt.subplots(figsize=(5, 3.8))
        palette = [AMBER, BLUE, GREEN, "#c084fc", "#f87171", "#67e8f9"]
        for i, row in bench_df.reset_index(drop=True).iterrows():
            clr = palette[i % len(palette)]
            ax.scatter(row["Accuracy"], row["ROC-AUC"], color=clr, s=120, zorder=5)
            ax.annotate(row["Model"], (row["Accuracy"], row["ROC-AUC"]),
                        textcoords="offset points", xytext=(8, 2), fontsize=8, color=clr)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Model Performance Landscape")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Multi-Metric Bar Chart**")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x     = np.arange(len(bench_df))
    width = 0.13
    metrics_colors = [
        ("Accuracy",  AMBER),
        ("ROC-AUC",   BLUE),
        ("Avg Prec",  GREEN),
        ("Precision", "#c084fc"),
        ("Recall",    "#f87171"),
        ("F1 Score",  "#67e8f9"),
    ]
    for i, (metric, clr) in enumerate(metrics_colors):
        ax.bar(x + i * width, bench_df[metric], width,
               label=metric, color=clr, alpha=0.85, edgecolor="none")
    ax.set_xticks(x + width * (len(metrics_colors) - 1) / 2)
    ax.set_xticklabels(bench_df["Model"], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("All Models · All Metrics", pad=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()
