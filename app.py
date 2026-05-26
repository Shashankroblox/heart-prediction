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

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
    [data-testid="stSidebar"] label { color: #94a3b8 !important; }
    hr { border-color: #334155 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #ef4444, #dc2626) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; padding: 12px 28px !important;
        font-size: 15px !important; font-weight: 700 !important; width: 100%;
    }
    [data-testid="metric-container"] {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 10px; padding: 14px !important;
    }
    [data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 12px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
    .risk-card {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
    }
    .section-hdr {
        font-size: 10px; font-weight: 700; color: #475569;
        letter-spacing: 0.1em; text-transform: uppercase;
        border-bottom: 1px solid #334155; padding-bottom: 6px; margin: 18px 0 10px;
    }
    .badge {
        display:inline-block; background:rgba(16,185,129,.15);
        border:1px solid rgba(16,185,129,.3); color:#10b981;
        border-radius:999px; padding:3px 12px; font-size:12px; font-weight:600;
    }
    .note { font-size:11px; color:#475569; margin-top:18px;
        padding:10px 14px; background:#1e293b; border-radius:8px;
        border-left:3px solid #334155; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# DATASET — self-contained inside cached function to avoid
# global-reference bugs on Streamlit Cloud
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    raw = np.array([
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

    X = raw[:, :13]
    y = raw[:, 13]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver="lbfgs")
    clf.fit(X_scaled, y)
    acc = float(round(clf.score(X_scaled, y) * 100, 1))
    n_samples = int(len(raw))
    return clf, scaler, acc, n_samples

model, scaler, accuracy, n_samples = train_model()

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
FEAT_LABELS = [
    "Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol",
    "Fasting BS", "Resting ECG", "Max HR", "Exercise Angina",
    "ST Depression", "ST Slope", "Vessels", "Thalassemia",
]

# ──────────────────────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────────────────────
def make_gauge(pct):
    pct = float(pct)
    if pct < 35:
        color, label = "#10b981", "LOW RISK"
    elif pct < 65:
        color, label = "#f59e0b", "MODERATE RISK"
    else:
        color, label = "#ef4444", "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 48, "color": color}},
        title={"text": label, "font": {"size": 13, "color": color}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "#334155",
                "tickfont": {"color": "#475569", "size": 10},
            },
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#0f172a",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],   "color": "rgba(16,185,129,0.1)"},
                {"range": [35, 65],  "color": "rgba(245,158,11,0.1)"},
                {"range": [65, 100], "color": "rgba(239,68,68,0.1)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        margin={"l": 20, "r": 20, "t": 50, "b": 10},
        height=240,
    )
    return fig


def make_coef_chart():
    coefs = [float(c) for c in model.coef_[0]]
    colors = ["#ef4444" if c > 0 else "#10b981" for c in coefs]
    fig = go.Figure(go.Bar(
        x=coefs,
        y=FEAT_LABELS,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        font={"color": "#94a3b8", "size": 11},
        xaxis={"gridcolor": "#334155", "zerolinecolor": "#475569"},
        yaxis={"gridcolor": "#334155"},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=340,
    )
    return fig


def make_contrib_chart(features_scaled):
    coefs = model.coef_[0]
    fvals = features_scaled[0]
    contribs = coefs * fvals
    # convert to plain Python floats — avoids numpy type issues in Plotly
    abs_contribs = [abs(float(c)) for c in contribs]
    order = sorted(range(13), key=lambda i: abs_contribs[i], reverse=True)[:8]
    labels = [FEAT_LABELS[i] for i in order]
    vals   = [float(contribs[i]) for i in order]
    colors = ["#ef4444" if v > 0 else "#10b981" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals[::-1],
        y=labels[::-1],
        orientation="h",
        marker_color=colors[::-1],
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        font={"color": "#94a3b8", "size": 11},
        xaxis={"gridcolor": "#334155", "zerolinecolor": "#475569"},
        yaxis={"gridcolor": "#334155"},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=300,
    )
    return fig

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ CardioPredict AI")
    st.markdown(f'<div class="badge">✓ Trained · {accuracy}% accuracy</div>', unsafe_allow_html=True)
    st.markdown("")

    st.markdown('<div class="section-hdr">🧬 Demographics</div>', unsafe_allow_html=True)
    age     = st.slider("Age (years)", 20, 80, 55)
    sex     = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x else "Female")
    cp      = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                format_func=lambda x: {0:"Typical angina",1:"Atypical angina",
                                        2:"Non-anginal pain",3:"Asymptomatic"}[x])

    st.markdown('<div class="section-hdr">💉 Resting Vitals</div>', unsafe_allow_html=True)
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 130)
    chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
    fbs      = st.selectbox("Fasting Blood Sugar", [0, 1],
                 format_func=lambda x: "> 120 mg/dl" if x else "≤ 120 mg/dl")
    restecg  = st.selectbox("Resting ECG", [0, 1, 2],
                 format_func=lambda x: {0:"Normal",1:"ST-T abnormality",2:"LV hypertrophy"}[x])

    st.markdown('<div class="section-hdr">🏃 Exercise Test</div>', unsafe_allow_html=True)
    thalach = st.slider("Max Heart Rate (bpm)", 60, 220, 150)
    exang   = st.selectbox("Exercise-Induced Angina", [0, 1],
                format_func=lambda x: "Yes" if x else "No")
    oldpeak = st.slider("ST Depression (mm)", 0.0, 7.0, 1.0, step=0.1)
    slope   = st.selectbox("ST Slope", [0, 1, 2],
                format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])

    st.markdown('<div class="section-hdr">🔬 Advanced Diagnostics</div>', unsafe_allow_html=True)
    ca   = st.selectbox("Major Vessels (fluoroscopy)", [0, 1, 2, 3],
             format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''}")
    thal = st.selectbox("Thalassemia", [1, 2, 3],
             format_func=lambda x: {1:"Normal",2:"Fixed defect",3:"Reversible defect"}[x])

    st.markdown("")
    predict_btn = st.button("❤️  Analyse Risk")

# ──────────────────────────────────────────────────────────────
# MAIN PANEL
# ──────────────────────────────────────────────────────────────
st.markdown("## CardioPredict AI")
st.markdown(
    "Heart disease risk predictor trained on the **Cleveland Heart Disease Dataset** "
    "(UCI ML Repository) · Logistic Regression"
)
st.divider()

if not predict_btn:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Samples", n_samples)
    c2.metric("Features", 13)
    c3.metric("Algorithm", "Logistic Reg.")
    c4.metric("Train Accuracy", f"{accuracy}%")

    st.markdown("")
    st.markdown("### Model Feature Coefficients")
    st.caption("🔴 Red = increases risk   🟢 Green = reduces risk")
    st.plotly_chart(make_coef_chart(), use_container_width=True)

    st.markdown(
        '<div class="note">⚠️ Educational tool only. Not a substitute for medical advice.</div>',
        unsafe_allow_html=True,
    )

else:
    # Build feature vector — explicit Python floats throughout
    feat = np.array([[
        float(age), float(sex), float(cp), float(trestbps), float(chol),
        float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak),
        float(slope), float(ca), float(thal),
    ]])
    feat_scaled = scaler.transform(feat)
    prob = float(model.predict_proba(feat_scaled)[0][1])
    pct  = round(prob * 100, 1)

    if pct < 35:
        risk_color, risk_label, risk_emoji = "#10b981", "Low Risk", "✅"
    elif pct < 65:
        risk_color, risk_label, risk_emoji = "#f59e0b", "Moderate Risk", "⚠️"
    else:
        risk_color, risk_label, risk_emoji = "#ef4444", "High Risk", "🚨"

    interp = {
        "Low Risk":      "Low probability of heart disease. Continue heart-healthy habits.",
        "Moderate Risk": "Moderate risk factors identified. Clinical evaluation advised.",
        "High Risk":     "Significant risk markers detected. Prompt medical evaluation strongly recommended.",
    }[risk_label]

    # ── Gauge + Profile ──
    col_g, col_p = st.columns(2)

    with col_g:
        st.plotly_chart(make_gauge(pct), use_container_width=True)
        st.markdown(
            f'<div class="risk-card" style="border-left:3px solid {risk_color}">'
            f'<strong style="color:{risk_color}">{risk_emoji} {risk_label}</strong><br>'
            f'<span style="color:#94a3b8;font-size:13px">{interp}</span></div>',
            unsafe_allow_html=True,
        )

    with col_p:
        st.markdown("#### 👤 Patient Profile")
        a1, a2, a3 = st.columns(3)
        a1.metric("Age", f"{age} yrs")
        a2.metric("Sex", "Male" if sex else "Female")
        a3.metric("Max HR", f"{thalach} bpm")
        b1, b2, b3 = st.columns(3)
        b1.metric("Cholesterol", f"{chol}")
        b2.metric("BP", f"{trestbps}")
        b3.metric("Vessels", f"{ca}")

        # Clinical signals
        signals = []
        if ca > 0:         signals.append(f"{ca} blocked vessel{'s' if ca>1 else ''} on fluoroscopy")
        if thal == 3:      signals.append("Reversible thalassemia defect")
        if oldpeak > 2:    signals.append(f"High ST depression ({oldpeak} mm)")
        if exang == 1:     signals.append("Exercise-induced angina")
        if cp == 3:        signals.append("Asymptomatic chest pain pattern")
        if thalach < 120:  signals.append(f"Low max heart rate ({thalach} bpm)")
        if chol > 300:     signals.append(f"Elevated cholesterol ({chol} mg/dl)")
        if trestbps > 160: signals.append(f"High resting BP ({trestbps} mmHg)")

        if signals:
            st.markdown("")
            st.markdown("#### 🔍 Key Signals")
            for s in signals[:4]:
                st.markdown(
                    f'<div class="risk-card">'
                    f'<span style="color:{risk_color};margin-right:8px">●</span>'
                    f'<span style="color:#cbd5e1;font-size:13px">{s}</span></div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Recommendations + Contributions ──
    col_r, col_f = st.columns(2)

    with col_r:
        st.markdown("#### 📋 Recommendations")
        recs = {
            "Low Risk":      ["Maintain healthy lifestyle & routine check-ups",
                              "Continue regular aerobic exercise",
                              "Monitor cholesterol annually"],
            "Moderate Risk": ["Consult a cardiologist for evaluation",
                              "Adopt heart-healthy diet (low saturated fat)",
                              "Exercise with medical guidance",
                              "Monitor BP & cholesterol regularly"],
            "High Risk":     ["Seek urgent cardiology consultation",
                              "Further diagnostic testing recommended",
                              "Review medications with your physician",
                              "Immediate lifestyle changes are critical"],
        }[risk_label]
        for r in recs:
            st.markdown(
                f'<div class="risk-card">'
                f'<span style="color:{risk_color};font-weight:700;margin-right:8px">→</span>'
                f'<span style="color:#cbd5e1;font-size:13px">{r}</span></div>',
                unsafe_allow_html=True,
            )

    with col_f:
        st.markdown("#### 📊 Feature Contributions")
        st.caption("How each input pushed the score up or down")
        st.plotly_chart(make_contrib_chart(feat_scaled), use_container_width=True)

    st.markdown(
        '<div class="note">⚠️ Educational demonstration only. '
        'Does not constitute medical advice. Consult a qualified physician.</div>',
        unsafe_allow_html=True,
    )
