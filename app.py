"""
============================================================
 Lab 5.0 – Student Grade Risk Predictor
 Artificial Intelligence 5.0
============================================================
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder
from sklearn.pipeline         import Pipeline
from sklearn.impute            import SimpleImputer
from sklearn.ensemble         import RandomForestClassifier

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Grade Risk Predictor",
    page_icon="🎓",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        border-radius: 16px; padding: 2rem 2.5rem;
        color: white; text-align: center; margin-bottom: 2rem;
    }
    .hero h1 { font-size: 2rem; font-weight: 800; margin: 0; }
    .hero p  { font-size: 0.95rem; opacity: 0.85; margin: 0.4rem 0 0; }

    .card {
        background: #f8f9ff; border-radius: 14px;
        padding: 1.5rem 1.8rem; margin-bottom: 1.2rem;
        border: 1px solid #e3eaf7;
    }
    .card-title {
        font-size: 1rem; font-weight: 700; color: #1a73e8;
        margin-bottom: 1rem; text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .result-box {
        border-radius: 16px; padding: 2rem;
        text-align: center; margin-top: 1rem;
    }
    .result-high   { background:#fff5f5; border:2px solid #e53935; }
    .result-medium { background:#fff8f0; border:2px solid #fb8c00; }
    .result-low    { background:#f0fff4; border:2px solid #43a047; }

    .result-label {
        font-size: 2rem; font-weight: 800; margin-bottom: 0.3rem;
    }
    .label-high   { color: #e53935; }
    .label-medium { color: #fb8c00; }
    .label-low    { color: #43a047; }

    .gwa-display {
        font-size: 1.1rem; color: #555; margin-bottom: 1rem;
    }
    .gwa-value { font-weight: 700; color: #1a73e8; font-size: 1.3rem; }

    .conf-bar-wrap {
        background: #e9ecef; border-radius: 20px;
        height: 14px; overflow: hidden; margin: 4px 0 10px;
    }
    .conf-bar {
        height: 100%; border-radius: 20px;
        transition: width 0.5s ease;
    }
    .conf-label {
        display:flex; justify-content:space-between;
        font-size: 0.82rem; color: #555;
    }

    .tip-box {
        border-radius: 12px; padding: 1rem 1.2rem;
        font-size: 0.9rem; margin-top: 1rem;
    }
    .tip-high   { background:#fff0f0; border-left:4px solid #e53935; color:#b71c1c; }
    .tip-medium { background:#fff8f0; border-left:4px solid #fb8c00; color:#e65100; }
    .tip-low    { background:#f0fff4; border-left:4px solid #43a047; color:#1b5e20; }

    .weight-chip {
        display:inline-block; background:#e8f0fe; color:#1a73e8;
        border-radius:20px; padding:2px 10px; font-size:0.78rem;
        font-weight:600; margin-left:6px;
    }
    .stButton>button {
        background: linear-gradient(135deg,#1a73e8,#0d47a1);
        color:white; border:none; border-radius:10px;
        padding:0.7rem 2rem; font-size:1rem; font-weight:700;
        width:100%; cursor:pointer; transition:opacity 0.2s;
    }
    .stButton>button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
FEATURES     = ["Attendance", "Quiz_Avg", "Lab_Avg", "Midterm", "Final"]
WEIGHTS      = {"Attendance": 0.10, "Quiz_Avg": 0.20,
                "Lab_Avg": 0.20, "Midterm": 0.25, "Final": 0.25}
RISK_COLORS  = {"High Risk": "#e53935", "Medium Risk": "#fb8c00", "Low Risk": "#43a047"}
RISK_CLASS   = {"High Risk": "high",    "Medium Risk": "medium",  "Low Risk": "low"}
RISK_TIPS    = {
    "High Risk":   "⚠️ This student is at serious academic risk. Immediate intervention is recommended — consider tutoring, counseling, and closer monitoring of attendance and exam performance.",
    "Medium Risk": "🔔 This student shows moderate risk. Encourage consistent study habits, improve quiz performance, and maintain attendance above 80% to move into Low Risk.",
    "Low Risk":    "✅ This student is performing well! Keep up the great work. Continue consistent attendance and study habits to maintain this standing.",
}

def compute_gwa(scores: dict) -> float:
    return sum(scores[f] * WEIGHTS[f] for f in FEATURES)

def assign_risk(gwa: float) -> str:
    if gwa >= 85: return "Low Risk"
    if gwa >= 75: return "Medium Risk"
    return "High Risk"

# ════════════════════════════════════════════════════════════════════════════
# LOAD & TRAIN  (cached — runs once)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Training models, please wait…")
def load_and_train():
    df = pd.read_csv("dataset.csv")
    for col in FEATURES:
        df[col] = df[col].fillna(df[col].median())
    df["GWA"] = df.apply(
        lambda r: sum(r[f] * WEIGHTS[f] for f in FEATURES), axis=1)
    df["Risk_Level"] = df["GWA"].apply(assign_risk)

    X  = df[FEATURES]
    le = LabelEncoder()
    y  = le.fit_transform(df["Risk_Level"])
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    models = {
        "Random Forest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]),
    }
    for pipe in models.values():
        pipe.fit(X_train, y_train)

    return models, le

try:
    MODELS, LE = load_and_train()
except FileNotFoundError:
    st.error("❌ `dataset.csv` not found. Place it in the same folder as `app.py`.")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🎓 Student Grade Risk Predictor</h1>
  <p>Artificial Intelligence 5.0 — Lab 5.0 &nbsp;|&nbsp; ML Classification System</p>
</div>
""", unsafe_allow_html=True)

# ── Score inputs ──────────────────────────────────────────────────────────────
st.markdown('<div class="card-title">📝 Enter Student Scores</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    attendance = st.number_input(
        "Attendance (%)  ", min_value=0, max_value=100, value=75, step=1,
        help="Weight: 10%")
    st.caption("Weight: **10%**")

    lab_avg = st.number_input(
        "Laboratory Average  ", min_value=0, max_value=100, value=75, step=1,
        help="Weight: 20%")
    st.caption("Weight: **20%**")

    final = st.number_input(
        "Final Exam  ", min_value=0, max_value=100, value=75, step=1,
        help="Weight: 25%")
    st.caption("Weight: **25%**")

with col2:
    quiz_avg = st.number_input(
        "Quiz Average  ", min_value=0, max_value=100, value=75, step=1,
        help="Weight: 20%")
    st.caption("Weight: **20%**")

    midterm = st.number_input(
        "Midterm Exam  ", min_value=0, max_value=100, value=75, step=1,
        help="Weight: 25%")
    st.caption("Weight: **25%**")

# ── Live GWA preview ─────────────────────────────────────────────────────────
scores = {"Attendance": attendance, "Quiz_Avg": quiz_avg,
          "Lab_Avg": lab_avg, "Midterm": midterm, "Final": final}
gwa = compute_gwa(scores)

st.markdown(
    f'<div class="gwa-display">Computed GWA: '
    f'<span class="gwa-value">{gwa:.2f}</span>'
    f' &nbsp;/&nbsp; 100</div>',
    unsafe_allow_html=True
)

# GWA progress bar
bar_color = RISK_COLORS[assign_risk(gwa)]
st.markdown(
    f"<div style='background:#e9ecef;border-radius:20px;height:12px;overflow:hidden;margin-bottom:1.2rem'>"
    f"<div style='width:{min(gwa,100):.1f}%;background:{bar_color};"
    f"height:100%;border-radius:20px'></div></div>",
    unsafe_allow_html=True
)

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔮 Predict Risk Level")

if predict_clicked:
    pipe         = MODELS["Random Forest"]
    input_df     = pd.DataFrame([scores])
    pred_encoded = pipe.predict(input_df)[0]
    pred_proba   = pipe.predict_proba(input_df)[0]
    pred_label   = LE.inverse_transform([pred_encoded])[0]
    risk_cls     = RISK_CLASS[pred_label]
    risk_color   = RISK_COLORS[pred_label]

    # ── Result card ───────────────────────────────────────────────────────
    st.markdown(
        f'<div class="result-box result-{risk_cls}">'
        f'<div class="result-label label-{risk_cls}">{pred_label}</div>'
        f'<div style="color:#555;font-size:0.95rem">using <b>Random Forest</b> &nbsp;|&nbsp; GWA: <b>{gwa:.2f}</b></div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Confidence bars ───────────────────────────────────────────────────
    st.markdown("<br>**Prediction Confidence:**", unsafe_allow_html=True)
    for cls, prob in zip(LE.classes_, pred_proba):
        c = RISK_COLORS[cls]
        st.markdown(
            f'<div class="conf-label"><span style="color:{c};font-weight:600">{cls}</span>'
            f'<span>{prob:.1%}</span></div>'
            f'<div class="conf-bar-wrap">'
            f'<div class="conf-bar" style="width:{prob*100:.1f}%;background:{c}"></div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Recommendation tip ────────────────────────────────────────────────
    tip_cls = risk_cls
    st.markdown(
        f'<div class="tip-box tip-{tip_cls}">{RISK_TIPS[pred_label]}</div>',
        unsafe_allow_html=True
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.caption("Lab 5.0 · AI 5.0 · Student Academic Risk Classification · Random Forest | Decision Tree | Logistic Regression")
