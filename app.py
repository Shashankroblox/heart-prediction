import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
    [data-testid="stSidebar"] * { color: #f1f5f9 !important; }

    /* Inputs */
    .stSelectbox > div > div, .stNumberInput > div > div > input,
    .stSlider > div { background-color: #1e293b !important; color: #f1f5f9 !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px !important;
    }
    [data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 12px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 20px !important; }

    /* Divider */
    hr { border-color: #334155 !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #ef4444, #dc2626) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        width: 100%;
        box-shadow: 0 4px 20px rgba(239,68,68,0.3);
    }
    .stButton > button:hover { opacity: 0.9 !important; }

    /* Section headers */
    .section-header {
        font-size: 11px;
        font-weight: 700;
        color: #64748b;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 20px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #334155;
    }

    /* Risk cards */
    .risk-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .signal-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .rec-arrow { font-weight: 700; margin-right: 8px; }

    /* Disclaimer */
    .disclaimer {
        font-size: 11px;
        color: #475569;
        margin-top: 20px;
        padding: 10px 14px;
        background: #1e293b;
        border-radius: 8px;
        border-left: 3px solid #334155;
    }

    /* Accuracy badge */
    .accuracy-badge {
        display: inline-block;
        background: rgba(16,185,129,0.15);
        border: 1px solid rgba(16,185,129,0.3);
        color: #10b981;
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# DATASET  (Cleveland Heart Disease – UCI ML Repository)
# age, sex, cp, trestbps, chol, fbs, restecg, thalach,
# exang, oldpeak, slope, ca, thal, target
# ──────────────────────────────────────────────────────────────
RAW = np.array([
    [63,1,3,145,233,1,0,150,0,2.3,0,0,1,0],[37,1,2,130,250,0,1,187,0,3.5,0,0,2,0],
    [41,0,1,130,204,0,0,172,0,1.4,2,0,2,0],[56,1,1,120,236,0,1,178,0,0.8,2,0,2,0],
    [57,0,0,120,354,0,1,163,1,0.6,2,0,2,0],[57,1,0,140,192,0,1,148,0,0.4,1,0,1,0],
    [56,0,1,140,294,0,0,153,0,1.3,1,0,2,0],[44,1,1,120,263,0,1,173,0,0.0,2,0,3,0],
    [52,1,2,172,199,1,1,162,0,0.5,2,0,3,0],[57,1,2,150,168,0,1,174,0,1.6,2,0,2,0],
    [54,1,0,140,239,0,1,160,0,1.2,2,0,2,0],[48,0,2,130,275,0,1,139,0,0.2,2,0,2,0],
    [49,1,1,130,266,0,1,171,0,0.6,2,0,2,0],[64,1,3,110,211,0,0,144,1,1.8,1,0,2,0],
    [58,0,2,150,283,1,0,162,0,1.0,2,0,2,0],[50,0,2,120,219,0,1,158,0,1.6,1,0,2,0],
    [58,0,2,120,340,0,1,172,0,0.0,2,0,2,0],[66,0,3,150,226,0,1,114,0,2.6,0,0,2,0],
    [43,1,0,150,247,0,1,171,0,1.5,2,0,2,0],[69,0,3,140,239,0,1,151,0,1.8,2,2,2,0],
    [59,1,0,135,234,0,1,161,0,0.5,1,0,3,0],[44,1,2,130,233,0,1,179,1,0.4,2,0,2,0],
    [42,1,0,140,226,0,1,178,0,0.0,2,0,2,0],[61,1,2,150,243,1,1,137,1,1.0,1,0,2,0],
    [40,1,3,140,199,0,1,178,1,1.4,2,0,3,0],[71,0,1,160,302,0,1,162,0,0.4,2,2,2,0],
    [51,1,2,110,175,0,1,123,0,0.6,2,0,2,0],[65,0,2,140,417,1,0,157,0,0.8,2,1,2,0],
    [53,0,2,130,197,1,0,152,0,1.2,0,0,2,0],[41,0,1,105,198,0,1,168,0,0.0,2,1,2,0],
    [66,1,0,120,302,0,0,151,0,0.4,1,0,2,0],[62,0,2,140,394,0,0,157,0,1.2,1,0,2,0],
    [55,1,3,125,212,0,1,168,0,1.0,2,1,2,0],[34,1,3,118,182,0,0,174,0,0.0,2,0,2,0],
    [48,1,0,122,222,0,0,186,0,0.0,2,0,2,0],[55,0,0,132,342,0,1,166,0,1.2,2,0,2,0],
    [39,1,2,120,339,0,1,170,0,0.0,2,0,2,0],[45,0,1,130,237,0,0,170,0,0.0,2,0,2,0],
    [54,0,2,160,201,0,1,163,0,0.0,2,1,2,0],[29,1,1,130,204,0,0,202,0,0.0,2,0,2,0],
    [43,0,2,122,213,0,1,165,0,0.2,1,0,2,0],[54,1,2,125,216,0,0,140,0,0.0,1,0,1,0],
    [67,1,0,160,286,0,0,108,1,1.5,1,3,2,1],[67,1,0,120,229,0,0,129,1,2.6,1,2,3,1],
    [62,0,0,140,268,0,0,160,0,3.6,0,2,2,1],[63,1,0,130,254,0,0,147,0,1.4,1,1,3,1],
    [53,1,0,140,203,1,0,155,1,3.1,0,0,3,1],[55,1,0,132,353,0,1,132,1,1.2,1,1,3,1],
    [65,0,0,150,225,0,0,114,0,1.0,1,3,3,1],[56,1,0,130,256,1,0,142,1,0.6,1,1,1,1],
    [54,1,0,125,273,0,0,152,0,0.5,0,1,2,1],[49,1,0,130,266,0,1,171,0,0.6,2,0,2,1],
    [64,1,0,110,211,0,0,144,1,1.8,1,0,2,1],[70,1,0,145,174,0,1,125,1,2.6,0,0,3,1],
    [61,1,0,148,203,0,1,161,0,0.0,2,1,3,1],[71,0,0,112,149,0,1,125,0,1.6,1,0,2,1],
    [74,0,1,120,269,0,0,121,1,0.2,2,1,2,1],[68,0,2,120,211,0,0,115,0,1.5,1,0,2,1],
    [57,1,0,130,131,0,1,115,1,1.2,1,1,3,1],[52,1,0,108,233,1,1,147,0,0.1,2,3,3,1],
    [51,0,0,130,305,0,1,142,1,1.2,1,0,3,1],[68,1,2,180,274,1,0,150,1,1.6,1,0,3,1],
    [43,1,0,120,177,0,0,120,1,2.5,1,0,3,1],[55,0,0,180,327,0,2,117,1,3.4,1,0,2,1],
    [62,1,0,130,231,0,1,146,0,1.8,1,3,3,1],[71,1,2,160,302,0,1,162,0,0.4,2,2,2,1],
    [46,1,2,140,311,0,1,120,1,1.8,1,2,3,1],[66,1,0,160,246,0,1,120,1,0.0,1,3,1,1],
    [58,1,2,125,300,0,0,171,0,0.0,2,2,3,1],[60,1,0,140,293,0,0,170,0,1.2,1,2,3,1],
    [65,1,0,110,248,0,0,158,0,0.6,2,2,1,1],[55,1,0,140,217,0,1,111,1,5.6,0,0,3,1],
    [54,1,0,122,286,0,0,116,1,3.2,1,2,2,1],[44,1,0,120,169,0,1,144,1,2.8,0,0,1,1],
    [51,1,0,140,299,0,1,173,1,1.6,2,0,3,1],[51,1,2,94,227,0,1,154,1,0.0,2,1,3,1],
    [59,1,2,126,218,1,1,134,0,2.2,1,1,1,1],[57,0,0,140,241,0,1,123,1,0.2,1,0,3,1],
    [45,1,0,104,208,0,0,148,1,3.0,1,0,2,1],[60,0,2,102,318,0,1,160,0,0.0,2,1,2,1],
    [55,1,0,160,289,0,0,145,1,0.8,1,1,3,1],[56,1,0,150,213,1,0,125,1,1.0,1,0,2,1],
    [63,1,0,150,223,0,0,115,0,0.0,2,0,1,0],[52,1,0,128,205,1,1,184,0,0.0,2,0,2,0],
    [55,0,0,128,205,0,2,130,1,2.0,1,1,3,1],[47,1,0,110,275,0,0,118,1,1.0,1,1,2,1],
    [46,1,0,120,230,0,1,150,0,0.0,2,0,2,0],[54,0,2,108,267,0,0,167,0,0.0,2,0,2,0],
    [54,1,0,110,208,0,0,142,0,0.0,2,0,2,0],[66,0,2,178,228,1,1,165,1,1.0,1,2,3,1],
    [58,1,0,100,234,0,1,156,0,0.1,2,1,3,1],[60,1,0,145,282,0,0,142,1,2.8,1,2,3,1],
], dtype=float)

X_raw = RAW[:, :13]
y = RAW[:, 13]

# ──────────────────────────────────────────────────────────────
# TRAIN MODEL  (cached so it only runs once per session)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X, y)
    acc = model.score(X, y)
    return model, scaler, round(acc * 100, 1)

model, scaler, accuracy = train_model()

# ──────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE CHART
# ──────────────────────────────────────────────────────────────
FEAT_LABELS = {
    "age": "Age", "sex": "Sex", "cp": "Chest Pain Type",
    "trestbps": "Resting BP", "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
    "thalach": "Max Heart Rate", "exang": "Exercise Angina",
    "oldpeak": "ST Depression", "slope": "ST Slope",
    "ca": "Major Vessels", "thal": "Thalassemia",
}
FEAT_KEYS = list(FEAT_LABELS.keys())

def feature_importance_chart():
    coefs = model.coef_[0]
    labels = list(FEAT_LABELS.values())
    colors = ["#ef4444" if c > 0 else "#10b981" for c in coefs]
    fig = go.Figure(go.Bar(
        x=coefs,
        y=labels,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Feature Coefficients", font=dict(color="#94a3b8", size=13)),
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font=dict(color="#94a3b8", size=11),
        xaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
        yaxis=dict(gridcolor="#334155"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=340,
    )
    return fig

def gauge_chart(pct):
    if pct < 35:
        color, label = "#10b981", "LOW RISK"
    elif pct < 65:
        color, label = "#f59e0b", "MODERATE RISK"
    else:
        color, label = "#ef4444", "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 44, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#475569"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "#10b98120"},
                {"range": [35, 65], "color": "#f59e0b20"},
                {"range": [65, 100],"color": "#ef444420"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": pct,
            },
        },
        title={"text": label, "font": {"size": 14, "color": color}},
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        font=dict(color="#f1f5f9"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=260,
    )
    return fig

# ──────────────────────────────────────────────────────────────
# SIDEBAR — INPUT FORM
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ CardioPredict AI")
    st.markdown(
        f'<span class="accuracy-badge">✓ Model trained · {accuracy}% accuracy</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">🧬 Demographics</div>', unsafe_allow_html=True)
    age      = st.slider("Age", 20, 80, 55)
    sex      = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp       = st.selectbox("Chest Pain Type", [0,1,2,3],
                 format_func=lambda x: {0:"Typical angina",1:"Atypical angina",2:"Non-anginal",3:"Asymptomatic"}[x])

    st.markdown('<div class="section-header">💉 Resting Vitals</div>', unsafe_allow_html=True)
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 130)
    chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
    fbs      = st.selectbox("Fasting Blood Sugar", [0, 1],
                 format_func=lambda x: "> 120 mg/dl" if x == 1 else "≤ 120 mg/dl")
    restecg  = st.selectbox("Resting ECG", [0,1,2],
                 format_func=lambda x: {0:"Normal",1:"ST-T abnormality",2:"LV hypertrophy"}[x])

    st.markdown('<div class="section-header">🏃 Exercise Test</div>', unsafe_allow_html=True)
    thalach  = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, 150)
    exang    = st.selectbox("Exercise-Induced Angina", [0, 1],
                 format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak  = st.slider("ST Depression", 0.0, 7.0, 1.0, step=0.1)
    slope    = st.selectbox("ST Slope", [0,1,2],
                 format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])

    st.markdown('<div class="section-header">🔬 Advanced Diagnostics</div>', unsafe_allow_html=True)
    ca       = st.selectbox("Major Vessels (fluoroscopy)", [0,1,2,3],
                 format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}")
    thal     = st.selectbox("Thalassemia", [1,2,3],
                 format_func=lambda x: {1:"Normal",2:"Fixed defect",3:"Reversible defect"}[x])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("❤️  Analyse Risk")

# ──────────────────────────────────────────────────────────────
# MAIN PANEL
# ──────────────────────────────────────────────────────────────
st.markdown("## CardioPredict AI")
st.markdown("Heart disease risk predictor trained on the **Cleveland Heart Disease Dataset** (UCI ML Repository) using logistic regression.")
st.divider()

if not predict_btn:
    # Landing state — show model info + feature importance
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Samples", len(RAW))
    col2.metric("Features", 13)
    col3.metric("Algorithm", "Logistic Reg.")
    col4.metric("Accuracy", f"{accuracy}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Feature Importance")
    st.markdown("Red bars increase risk · Green bars are protective")
    st.plotly_chart(feature_importance_chart(), use_container_width=True)

    st.markdown(
        '<div class="disclaimer">⚠️ This tool is for educational demonstration only. '
        'It does not constitute medical advice. Always consult a qualified physician.</div>',
        unsafe_allow_html=True,
    )

else:
    # ── PREDICTION ──
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]], dtype=float)
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0][1]
    pct  = round(prob * 100, 1)

    if pct < 35:
        risk_color, risk_label, risk_emoji = "#10b981", "Low Risk", "✅"
    elif pct < 65:
        risk_color, risk_label, risk_emoji = "#f59e0b", "Moderate Risk", "⚠️"
    else:
        risk_color, risk_label, risk_emoji = "#ef4444", "High Risk", "🚨"

    # ── Row 1: Gauge + Patient Profile ──
    col_g, col_p = st.columns([1, 1])

    with col_g:
        st.plotly_chart(gauge_chart(pct), use_container_width=True)
        interp = {
            "Low Risk":      "The model predicts a low probability of heart disease. Continue heart-healthy habits.",
            "Moderate Risk": "The model identifies moderate risk factors. Clinical evaluation is advised.",
            "High Risk":     "The model detects significant risk markers. Prompt medical evaluation is strongly recommended.",
        }[risk_label]
        st.markdown(
            f'<div class="risk-card" style="border-left: 3px solid {risk_color};">'
            f'<b style="color:{risk_color}">{risk_emoji} {risk_label}</b><br>'
            f'<span style="color:#94a3b8;font-size:13px">{interp}</span></div>',
            unsafe_allow_html=True,
        )

    with col_p:
        st.markdown("#### Patient Profile")
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("Age", f"{age} yrs")
        r1c2.metric("Sex", "Male" if sex == 1 else "Female")
        r1c3.metric("Max HR", f"{thalach} bpm")

        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("Cholesterol", f"{chol} mg/dl")
        r2c2.metric("BP", f"{trestbps} mmHg")
        r2c3.metric("Vessels", ca)

        st.markdown("<br>", unsafe_allow_html=True)

        # Clinical signals
        signals = []
        if ca > 0:      signals.append(f"{ca} blocked vessel{'s' if ca > 1 else ''} on fluoroscopy")
        if thal == 3:   signals.append("Reversible thalassemia defect detected")
        if oldpeak > 2: signals.append(f"High ST depression ({oldpeak} mm)")
        if exang == 1:  signals.append("Exercise-induced angina present")
        if cp == 3:     signals.append("Asymptomatic chest pain pattern")
        if thalach < 120: signals.append(f"Low max heart rate ({thalach} bpm)")
        if chol > 300:  signals.append(f"Elevated cholesterol ({chol} mg/dl)")
        if trestbps > 160: signals.append(f"High resting BP ({trestbps} mm Hg)")

        if signals:
            st.markdown("#### 🔍 Key Signals Detected")
            for s in signals[:4]:
                st.markdown(
                    f'<div class="risk-card">'
                    f'<span class="signal-dot" style="background:{risk_color}"></span>'
                    f'<span style="color:#cbd5e1;font-size:13px">{s}</span></div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Row 2: Recommendations + Feature Contributions ──
    col_r, col_f = st.columns([1, 1])

    with col_r:
        st.markdown("#### 📋 Recommendations")
        recs = {
            "Low Risk": [
                "Maintain healthy lifestyle and routine check-ups",
                "Continue regular aerobic exercise",
                "Monitor cholesterol annually",
            ],
            "Moderate Risk": [
                "Consult a cardiologist for detailed evaluation",
                "Adopt heart-healthy diet (reduce saturated fats)",
                "Increase physical activity with medical guidance",
                "Monitor BP and cholesterol regularly",
            ],
            "High Risk": [
                "Seek urgent cardiology consultation",
                "Further diagnostic testing strongly recommended",
                "Review medications with your physician",
                "Lifestyle changes are critical",
            ],
        }[risk_label]

        for rec in recs:
            st.markdown(
                f'<div class="risk-card">'
                f'<span class="rec-arrow" style="color:{risk_color}">→</span>'
                f'<span style="color:#cbd5e1;font-size:13px">{rec}</span></div>',
                unsafe_allow_html=True,
            )

    with col_f:
        st.markdown("#### 📊 Feature Contributions")
        coefs = model.coef_[0]
        feat_vals = features_scaled[0]
        contribs = coefs * feat_vals
        sorted_idx = np.argsort(np.abs(contribs))[::-1][:8]
        labels_sorted = [list(FEAT_LABELS.values())[i] for i in sorted_idx]
        vals_sorted = [contribs[i] for i in sorted_idx]
        colors = ["#ef4444" if v > 0 else "#10b981" for v in vals_sorted]

        fig2 = go.Figure(go.Bar(
            x=vals_sorted[::-1],
            y=labels_sorted[::-1],
            orientation="h",
            marker_color=colors[::-1],
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ))
        fig2.update_layout(
            paper_bgcolor="#1e293b",
            plot_bgcolor="#1e293b",
            font=dict(color="#94a3b8", size=11),
            xaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
            yaxis=dict(gridcolor="#334155"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        '<div class="disclaimer">⚠️ This prediction is for educational demonstration only '
        'and does not constitute medical advice. Always consult a qualified physician.</div>',
        unsafe_allow_html=True,
    )
