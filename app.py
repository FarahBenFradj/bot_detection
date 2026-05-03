"""
app.py — Streamlit Bot Detection Web Interface
Run:  streamlit run app.py
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🤖 Bot Detection AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 1rem;
    margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.main-header h1 { color: white; text-align: center; margin: 0; font-size: 2.6rem; }
.main-header p  { color: #f0f0f0; text-align: center; margin: 0.5rem 0 0; font-size: 1.1rem; }
.bot-alert   { background: linear-gradient(135deg,#f093fb,#f5576c); color:white;
               padding:1.5rem; border-radius:.5rem; margin:1rem 0; }
.human-alert { background: linear-gradient(135deg,#4facfe,#00f2fe); color:white;
               padding:1.5rem; border-radius:.5rem; margin:1rem 0; }
.stButton>button {
    background: linear-gradient(135deg,#667eea,#764ba2);
    color:white; border:none; padding:.75rem 2rem;
    font-size:1.1rem; border-radius:.5rem; font-weight:600; width:100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>🤖 AI-Powered Bot Detection System</h1>
  <p>Deep Learning for Social Media Security · TwiBot-20 Dataset</p>
</div>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    """Load trained model + preprocessor from models/."""
    import sys; sys.path.insert(0, ".")
    from tensorflow import keras
    from src.preprocessing import BotDataPreprocessor

    model_path = Path("models/bot_detector_deep_mlp_best.h5")
    prep_path  = Path("models/preprocessor.pkl")

    model, preprocessor = None, None

    if model_path.exists():
        model = keras.models.load_model(str(model_path))

    if prep_path.exists():
        preprocessor = BotDataPreprocessor()
        preprocessor.load(str(prep_path))

    return model, preprocessor


model, preprocessor = load_assets()

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict_from_form(user_dict: dict) -> float:
    """Return bot probability. Uses loaded model if available, else heuristic."""
    if model is not None and preprocessor is not None:
        X    = preprocessor.transform(user_dict)
        prob = float(model.predict(X, verbose=0).flatten()[0])
        return prob
    # ── Heuristic fallback (demo mode) ────────────────────────────────────────
    score = 0.0
    followers = user_dict.get("followers_count", 0)
    following = user_dict.get("friends_count", 0)
    tweets    = user_dict.get("statuses_count", 0)
    age_days  = user_dict.get("account_age_days", 1)

    ratio = followers / max(following, 1)
    rate  = tweets / max(age_days, 1)

    if ratio < 0.1:          score += 0.30
    if rate  > 20:           score += 0.25
    if user_dict.get("default_profile_image"): score += 0.20
    if not user_dict.get("has_description"):   score += 0.15
    if user_dict.get("screen_name_has_digits"):score += 0.10

    return min(score, 0.99)


def gauge_chart(probability: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Bot Probability (%)", "font": {"size": 20}},
        gauge={
            "axis":  {"range": [0, 100]},
            "bar":   {"color": "darkblue"},
            "steps": [
                {"range": [0,  35], "color": "#4facfe"},
                {"range": [35, 65], "color": "#f0f0f0"},
                {"range": [65, 100],"color": "#f5576c"},
            ],
            "threshold": {"line": {"color": "red", "width": 4},
                          "thickness": 0.75, "value": 80},
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 Settings")

    model_choice = st.selectbox(
        "Detection Model",
        ["Deep MLP", "Transformer", "Attention MLP", "MLP Baseline"],
    )
    threshold = st.slider("Detection threshold", 0.30, 0.90, 0.50, 0.05,
                          help="Accounts above this probability are flagged as bots")

    st.markdown("---")
    st.subheader("📊 Model Performance")

    from config import MODEL_METRICS
    m = MODEL_METRICS[model_choice]
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m['accuracy']:.2%}")
    c2.metric("F1",       f"{m['f1']:.4f}")
    c3.metric("AUC",      f"{m['auc']:.4f}")

    if model is None:
        st.warning("⚠️ No trained model found in models/.\n\n"
                   "Run `python scripts/train.py` first, or the app "
                   "will use a heuristic demo mode.")
    else:
        st.success("✅ Model loaded")

    st.markdown("---")
    st.info("**Dataset:** TwiBot-20\n\n"
            "Request access at: shangbin@cs.washington.edu")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Single Account", "📊 Batch Analysis", "📈 Analytics", "📖 About"]
)

# ─────────────────────── Tab 1 — Single Account ───────────────────────────────
with tab1:
    st.header("Single Account Analysis")

    with st.form("account_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Profile")
            username     = st.text_input("Username (@handle)")
            followers    = st.number_input("Followers",        min_value=0, value=100)
            following    = st.number_input("Following",        min_value=0, value=200)
            tweets       = st.number_input("Total tweets",     min_value=0, value=500)
            created_date = st.date_input("Account created",
                                         value=datetime(2018, 1, 1))

        with col2:
            st.subheader("Details")
            verified      = st.checkbox("Verified account")
            default_img   = st.checkbox("Default profile image", value=True)
            description   = st.text_area("Bio / Description", height=80)
            has_url       = st.checkbox("Has profile URL")
            has_location  = st.checkbox("Has location set")

        submitted = st.form_submit_button("🔍 Analyse Account")

    if submitted:
        age_days = (datetime.now() - datetime(
            created_date.year, created_date.month, created_date.day)).days

        user_dict = {
            "followers_count":       followers,
            "friends_count":         following,
            "statuses_count":        tweets,
            "account_age_days":      age_days,
            "verified":              int(verified),
            "default_profile_image": int(default_img),
            "has_description":       int(bool(description)),
            "has_url":               int(has_url),
            "has_location":          int(has_location),
            "screen_name_length":    len(username),
            "screen_name_has_digits":int(any(c.isdigit() for c in username)),
            "followers_friends_ratio": followers / max(following, 1),
            "tweet_frequency":       tweets / max(age_days, 1),
        }

        prob   = predict_from_form(user_dict)
        is_bot = prob > threshold

        st.markdown("---")
        st.subheader("🎯 Detection Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.plotly_chart(gauge_chart(prob), use_container_width=True)

        with r2:
            if is_bot:
                st.markdown(f"""
                <div class="bot-alert">
                  <h2 style="margin:0;">⚠️ BOT DETECTED</h2>
                  <p style="margin:.5rem 0 0;font-size:1.2rem;">
                    Confidence: {prob*100:.1f}%
                  </p>
                </div>""", unsafe_allow_html=True)
                st.warning("This account shows suspicious bot-like patterns.")
            else:
                st.markdown(f"""
                <div class="human-alert">
                  <h2 style="margin:0;">✅ LIKELY HUMAN</h2>
                  <p style="margin:.5rem 0 0;font-size:1.2rem;">
                    Confidence: {(1-prob)*100:.1f}%
                  </p>
                </div>""", unsafe_allow_html=True)
                st.success("This account appears to be operated by a human.")

            st.markdown("### 🔍 Risk Factors")
            risk = []
            if followers / max(following, 1) < 0.1:  risk.append("Low followers/following ratio")
            if tweets / max(age_days, 1) > 20:        risk.append("Very high tweet frequency")
            if default_img:                            risk.append("Default profile image")
            if not description:                        risk.append("Empty bio")
            if any(c.isdigit() for c in username):    risk.append("Digits in username")

            for r in risk:  st.markdown(f"⚠️ {r}")
            if not risk:    st.markdown("✅ No major risk factors")

        with r3:
            st.markdown("### 📊 Account Stats")
            st.dataframe(pd.DataFrame({
                "Metric": ["Followers", "Following", "Tweets", "Account Age", "F/F Ratio"],
                "Value":  [f"{followers:,}", f"{following:,}", f"{tweets:,}",
                           f"{age_days} days",
                           f"{followers/max(following,1):.2f}"],
            }), hide_index=True, use_container_width=True)

# ─────────────────────── Tab 2 — Batch Analysis ───────────────────────────────
with tab2:
    st.header("Batch Analysis")
    st.info("Upload a JSON file with a list of user records (TwiBot-20 schema).")

    uploaded = st.file_uploader("Upload JSON", type=["json"])
    if uploaded:
        data = json.load(uploaded)
        if isinstance(data, dict): data = [data]
        st.write(f"Loaded {len(data)} accounts.")

        results = []
        bar = st.progress(0)
        for i, u in enumerate(data):
            uid  = u.get("ID", u.get("id", f"user_{i}"))
            prob = predict_from_form(u.get("profile", u))
            results.append({"user_id": uid, "bot_probability": prob,
                            "prediction": "Bot" if prob > threshold else "Human"})
            bar.progress((i + 1) / len(data))

        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)

        bot_count   = (df_res["prediction"] == "Bot").sum()
        human_count = len(df_res) - bot_count
        fig = go.Figure(go.Pie(
            labels=["Human", "Bot"], values=[human_count, bot_count],
            hole=0.4, marker_colors=["#4facfe", "#f5576c"]
        ))
        fig.update_layout(title="Detection Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)

        csv = df_res.to_csv(index=False).encode()
        st.download_button("⬇ Download results CSV", csv,
                           "bot_detection_results.csv", "text/csv")

# ─────────────────────── Tab 3 — Analytics ────────────────────────────────────
with tab3:
    st.header("📈 Detection Analytics")
    st.info("Run batch analysis first to populate real analytics. "
            "Showing placeholder data below.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Analysed", "1,234")
    c2.metric("Bots Detected",  "567")
    c3.metric("Detection Rate", "45.9%")
    c4.metric("Model Accuracy", "96.3%")

    fc1, fc2 = st.columns(2)
    with fc1:
        fig = go.Figure(go.Pie(
            labels=["Human","Bot"], values=[667, 567],
            hole=0.4, marker_colors=["#4facfe","#f5576c"]
        ))
        fig.update_layout(title="Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)
    with fc2:
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        fig2  = go.Figure(go.Scatter(
            x=dates, y=np.random.randint(10, 50, 30),
            mode="lines+markers", line=dict(color="#667eea", width=3)
        ))
        fig2.update_layout(title="Daily Detections (demo)", height=300)
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────── Tab 4 — About ───────────────────────────────────────
with tab4:
    st.header("📖 About")
    st.markdown("""
## How It Works

Our deep learning system analyses **60+ features** per user account:

**👤 Profile features** — account age, follower/following counts, verification status,
profile completeness, username patterns.

**📝 Content features** — tweet frequency, URL/mention/hashtag usage, retweet ratio,
lexical diversity, source diversity, engagement rate.

**⏰ Temporal features** — average time between tweets, night/weekend activity ratios,
posting consistency.

**🤖 Models available**

| Model | Architecture | Strength |
|-------|-------------|---------|
| MLP Baseline | 256→128→64 | Fast, interpretable |
| Deep MLP | Residual blocks | Handles complex patterns |
| Attention MLP | Soft feature gate | Explainable feature weights |
| Transformer | Multi-head self-attention | State-of-the-art |

## Dataset — TwiBot-20

229,573 users · 33M+ tweets · Expert-annotated labels  
Request access: **shangbin@cs.washington.edu**

## References

- Feng et al. (2021). *TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark.* CIKM 2021.
- Kudugunta & Ferrara (2018). *Deep Neural Networks for Bot Detection.* Information Sciences.
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:gray;padding:1.5rem;">
  <p><strong>⚖️ Disclaimer:</strong> For research and educational purposes only.</p>
  <p>Built with TensorFlow & Streamlit · TwiBot-20 Dataset</p>
</div>
""", unsafe_allow_html=True)