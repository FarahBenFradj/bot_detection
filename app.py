"""
app.py — Streamlit Bot Detection Web Interface
Run:  streamlit run app.py
"""

import json
import random
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🤖 Bot Detection AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
  <p>Deep Learning for Social Media Security · Cresci-17 Dataset</p>
</div>
""", unsafe_allow_html=True)

# ── Real metrics from training ────────────────────────────────────────────────
REAL_METRICS = {
    "Transformer":   {"accuracy": 0.9930, "f1": 0.9954, "auc": 0.9994},
    "Attention MLP": {"accuracy": 0.9912, "f1": 0.9942, "auc": 0.9993},
    "Deep MLP":      {"accuracy": 0.9884, "f1": 0.9923, "auc": 0.9988},
    "MLP Baseline":  {"accuracy": 0.9865, "f1": 0.9911, "auc": 0.9989},
}

MODEL_FILES = {
    "Transformer":   "models/bot_detector_transformer_best.keras",
    "Attention MLP": "models/bot_detector_attention_best.keras",
    "Deep MLP":      "models/bot_detector_deep_mlp_best.keras",
    "MLP Baseline":  "models/bot_detector_mlp_best.keras",
}

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets(model_name: str):
    import sys; sys.path.insert(0, ".")
    from tensorflow import keras
    from src.preprocessing import BotDataPreprocessor

    model_path = Path(MODEL_FILES[model_name])
    prep_path  = Path("models/preprocessor.pkl")

    model, preprocessor = None, None

    if model_path.exists():
        try:
            model = keras.models.load_model(str(model_path))
        except Exception as e:
            st.warning(f"Could not load model: {e}")

    if prep_path.exists():
        preprocessor = BotDataPreprocessor()
        preprocessor.load(str(prep_path))

    return model, preprocessor


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 Settings")

    model_choice = st.selectbox(
        "Detection Model",
        list(REAL_METRICS.keys()),
    )
    threshold = st.slider("Detection threshold", 0.30, 0.90, 0.50, 0.05)

    model, preprocessor = load_assets(model_choice)

    st.markdown("---")
    st.subheader("📊 Model Performance")
    m = REAL_METRICS[model_choice]
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m['accuracy']:.2%}")
    c2.metric("F1",       f"{m['f1']:.4f}")
    c3.metric("AUC",      f"{m['auc']:.4f}")

    st.markdown("---")
    if model is None:
        st.warning("⚠️ No trained model found.\nRun `python scripts/train.py` first.")
    else:
        st.success(f"✅ {model_choice} loaded")

    st.info("**Dataset:** Cresci-17\n\n14,368 labelled accounts")


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_user_dict(username, followers, following, tweets,
                    created_date, verified, default_img,
                    description, has_url, has_location):
    """Build a TwiBot/Cresci-compatible user dict from form inputs."""
    age_days = (datetime.now() - datetime(
        created_date.year, created_date.month, created_date.day)).days
    return {
        "ID": username or "form_user",
        "profile": {
            "screen_name":         username,
            "name":                username,
            "followers_count":     followers,
            "friends_count":       following,
            "statuses_count":      tweets,
            "favourites_count":    0,
            "listed_count":        0,
            "verified":            verified,
            "default_profile":     False,
            "default_profile_image": default_img,
            "geo_enabled":         False,
            "description":         description,
            "url":                 "http://example.com" if has_url else "",
            "location":            "somewhere" if has_location else "",
            "created_at":          datetime(
                created_date.year, created_date.month, created_date.day
            ).strftime("%a %b %d %H:%M:%S +0000 %Y"),
        },
        "tweet": [],
    }


def predict_user(user_dict: dict) -> float:
    """Return bot probability using model, or heuristic fallback."""
    if model is not None and preprocessor is not None:
        X    = preprocessor.transform_single(user_dict)
        return float(model.predict(X, verbose=0).flatten()[0])

    # Heuristic fallback (demo mode, no model loaded)
    p = user_dict.get("profile", user_dict)
    followers = p.get("followers_count", 0)
    following = p.get("friends_count", 0)
    tweets    = p.get("statuses_count", 0)
    age_days  = max((datetime.now() - datetime(2018, 1, 1)).days, 1)
    score = 0.0
    if followers / max(following, 1) < 0.1:           score += 0.30
    if tweets / max(age_days, 1) > 20:                score += 0.25
    if p.get("default_profile_image"):                score += 0.20
    if not p.get("description"):                      score += 0.15
    if any(c.isdigit() for c in str(p.get("screen_name", ""))): score += 0.10
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
                          "thickness": 0.75, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Single Account", "📊 Batch Analysis", "📈 Model Comparison", "📖 About"]
)

# ─────────────────────── Tab 1 — Single Account ───────────────────────────────
with tab1:
    st.header("Single Account Analysis")

    # ── Load sample data for random fill ─────────────────────────────────────
    @st.cache_data
    def load_sample_data():
        import json
        data_path = Path("data/cresci17.json")
        if not data_path.exists():
            # fallback: try splits
            data_path = Path("data/splits/test.json")
        if not data_path.exists():
            return [], []
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        humans = [u for u in data if str(u.get("label", "")) == "0"]
        bots   = [u for u in data if str(u.get("label", "")) == "1"]
        return humans, bots

    humans, bots = load_sample_data()

    # ── Random fill buttons ───────────────────────────────────────────────────
    st.markdown("**Quick fill from dataset:**")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])

    with col_btn1:
        gen_human = st.button("👤 Random Human", use_container_width=True)
    with col_btn2:
        gen_bot   = st.button("🤖 Random Bot",   use_container_width=True)

    # Pick a random user and store in session state
    if gen_human and humans:
        # Force clear previous state then pick a human
        st.session_state.pop("prefill", None)
        candidate = random.choice(humans)
        # Double-check it's really a human
        while str(candidate.get("label", "0")) != "0" and len(humans) > 1:
            candidate = random.choice(humans)
        st.session_state["prefill"] = candidate
        st.session_state["prefill_type"] = "human"
        st.rerun()

    if gen_bot and bots:
        # Force clear previous state then pick a bot
        st.session_state.pop("prefill", None)
        candidate = random.choice(bots)
        # Double-check it's really a bot
        while str(candidate.get("label", "1")) != "1" and len(bots) > 1:
            candidate = random.choice(bots)
        st.session_state["prefill"] = candidate
        st.session_state["prefill_type"] = "bot"
        st.rerun()

    # Extract prefill values
    prefill = st.session_state.get("prefill", None)
    if prefill:
        p            = prefill.get("profile", {})
        _username    = p.get("screen_name", "")
        _followers   = int(p.get("followers_count", 0))
        _following   = int(p.get("friends_count", 0))
        _tweets      = int(p.get("statuses_count", 0))
        _verified    = bool(p.get("verified", False))
        _default_img = bool(p.get("default_profile_image", True))
        _description = p.get("description", "") or ""
        _has_url     = bool(p.get("url", ""))
        _has_location= bool(p.get("location", ""))
        # Parse created_at
        _created = datetime(2018, 1, 1)
        raw_date = p.get("created_at", "")
        for fmt in ("%a %b %d %H:%M:%S +0000 %Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                _created = datetime.strptime(raw_date, fmt)
                break
            except (ValueError, TypeError):
                continue
        # Show a badge
        ptype = st.session_state.get("prefill_type", "")
        if ptype == "human":
            st.success(f"✅ Loaded a **Human** account from dataset: @{_username}")
        else:
            st.error(f"⚠️ Loaded a **Bot** account from dataset: @{_username}")
    else:
        _username = "testuser"; _followers = 100; _following = 200
        _tweets = 500; _verified = False; _default_img = True
        _description = ""; _has_url = False; _has_location = False
        _created = datetime(2018, 1, 1)

    # ── Form ──────────────────────────────────────────────────────────────────
    with st.form("account_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Profile")
            username     = st.text_input("Username (@handle)", value=_username)
            followers    = st.number_input("Followers",    min_value=0, value=_followers)
            following    = st.number_input("Following",    min_value=0, value=_following)
            tweets       = st.number_input("Total tweets", min_value=0, value=_tweets)
            created_date = st.date_input("Account created", value=_created)

        with col2:
            st.subheader("Details")
            verified     = st.checkbox("Verified account",        value=_verified)
            default_img  = st.checkbox("Default profile image",   value=_default_img)
            description  = st.text_area("Bio / Description",      value=_description, height=80)
            has_url      = st.checkbox("Has profile URL",         value=_has_url)
            has_location = st.checkbox("Has location set",        value=_has_location)

        submitted = st.form_submit_button("🔍 Analyse Account")

    if submitted:
        user_dict = build_user_dict(
            username, followers, following, tweets,
            created_date, verified, default_img,
            description, has_url, has_location
        )
        prob   = predict_user(user_dict)
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
                  <p style="margin:.5rem 0 0;font-size:1.2rem;">Confidence: {prob*100:.1f}%</p>
                </div>""", unsafe_allow_html=True)
                st.warning("This account shows suspicious bot-like patterns.")
            else:
                st.markdown(f"""
                <div class="human-alert">
                  <h2 style="margin:0;">✅ LIKELY HUMAN</h2>
                  <p style="margin:.5rem 0 0;font-size:1.2rem;">Confidence: {(1-prob)*100:.1f}%</p>
                </div>""", unsafe_allow_html=True)
                st.success("This account appears to be operated by a human.")

            st.markdown("### 🔍 Risk Factors")
            risk = []
            if followers / max(following, 1) < 0.1: risk.append("Low followers/following ratio")
            if tweets / max((datetime.now() - datetime(created_date.year, created_date.month, created_date.day)).days, 1) > 20:
                risk.append("Very high tweet frequency")
            if default_img:       risk.append("Default profile image")
            if not description:   risk.append("Empty bio")
            if any(c.isdigit() for c in username): risk.append("Digits in username")
            for r in risk: st.markdown(f"⚠️ {r}")
            if not risk:   st.markdown("✅ No major risk factors")

        with r3:
            age_days = (datetime.now() - datetime(created_date.year, created_date.month, created_date.day)).days
            st.markdown("### 📊 Account Stats")
            st.dataframe(pd.DataFrame({
                "Metric": ["Followers", "Following", "Tweets", "Account Age", "F/F Ratio"],
                "Value":  [f"{followers:,}", f"{following:,}", f"{tweets:,}",
                           f"{age_days} days", f"{followers/max(following,1):.2f}"],
            }), hide_index=True, use_container_width=True)
# ─────────────────────── Tab 2 — Batch Analysis ───────────────────────────────
with tab2:
    st.header("Batch Analysis")
    st.info("Upload a JSON file with a list of user records (Cresci-17 / TwiBot-20 schema).")

    uploaded = st.file_uploader("Upload JSON", type=["json"])
    if uploaded:
        data = json.load(uploaded)
        if isinstance(data, dict): data = [data]
        st.write(f"Loaded **{len(data)}** accounts.")

        results = []
        bar = st.progress(0)
        for i, u in enumerate(data):
            uid  = u.get("ID", u.get("id", f"user_{i}"))
            prob = predict_user(u)
            true_label = u.get("label", None)
            results.append({
                "user_id":        uid,
                "bot_probability": round(prob, 4),
                "prediction":     "Bot" if prob > threshold else "Human",
                "true_label":     ("Bot" if str(true_label) == "1" else "Human")
                                  if true_label is not None else "N/A",
            })
            bar.progress((i + 1) / len(data))

        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)

        bot_count   = (df_res["prediction"] == "Bot").sum()
        human_count = len(df_res) - bot_count

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Pie(
                labels=["Human", "Bot"], values=[human_count, bot_count],
                hole=0.4, marker_colors=["#4facfe", "#f5576c"]
            ))
            fig.update_layout(title="Prediction Distribution", height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Accuracy if labels available
            if df_res["true_label"].iloc[0] != "N/A":
                correct = (df_res["prediction"] == df_res["true_label"]).sum()
                acc = correct / len(df_res)
                st.metric("Accuracy on uploaded file", f"{acc:.2%}")
            st.metric("Bots detected",  f"{bot_count} ({bot_count/len(df_res)*100:.1f}%)")
            st.metric("Humans detected",f"{human_count} ({human_count/len(df_res)*100:.1f}%)")

        csv = df_res.to_csv(index=False).encode()
        st.download_button("⬇ Download results CSV", csv,
                           "bot_detection_results.csv", "text/csv")

# ─────────────────────── Tab 3 — Model Comparison ────────────────────────────
with tab3:
    st.header("📈 Model Comparison — Cresci-17")

    df_metrics = pd.DataFrame(REAL_METRICS).T.reset_index()
    df_metrics.columns = ["Model", "Accuracy", "F1-Score", "AUC-ROC"]
    df_metrics = df_metrics.sort_values("AUC-ROC", ascending=False)

    st.dataframe(
        df_metrics.style.highlight_max(subset=["Accuracy","F1-Score","AUC-ROC"],
                                        color="#d4edda"),
        use_container_width=True, hide_index=True
    )

    import plotly.express as px
    df_melt = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(df_melt, x="Metric", y="Score", color="Model", barmode="group",
                 range_y=[0.98, 1.0],
                 color_discrete_sequence=["#4C72B0","#DD8452","#55A868","#C44E52"],
                 title="Model Performance Comparison")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    results_dir = Path("results")
    for img_name in ["model_comparison.png", "model_radar.png"]:
        img_path = results_dir / img_name
        if img_path.exists():
            st.image(str(img_path), use_column_width=True)

# ─────────────────────── Tab 4 — About ───────────────────────────────────────
with tab4:
    st.header("📖 About")
    st.markdown("""
## How It Works

The system extracts **38 features** per user account across 4 categories:

**👤 Profile** — account age, follower/following counts, verification status, profile completeness, username patterns.

**📝 Content** — tweet frequency, URL/mention/hashtag usage, retweet ratio, lexical diversity, source diversity, engagement rate.

**⏰ Temporal** — average time between tweets, night/weekend/morning/afternoon activity ratios.

**🤖 Behavioural** — daily tweet rate, follower growth rate, reciprocity, network size.

## Models

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Transformer | 99.30% | 0.9954 | 0.9994 |
| Attention MLP | 99.12% | 0.9942 | 0.9993 |
| Deep MLP | 98.84% | 0.9923 | 0.9988 |
| MLP Baseline | 98.65% | 0.9911 | 0.9989 |

## Dataset — Cresci-17

14,368 labelled Twitter accounts · Bots and genuine users · Publicly available.

## References
- Cresci et al. (2017). *The Paradigm-Shift of Social Spambots.* WWW 2017.
- Kudugunta & Ferrara (2018). *Deep Neural Networks for Bot Detection.* Information Sciences.
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:gray;padding:1.5rem;">
  <p><strong>⚖️ Disclaimer:</strong> For research and educational purposes only.</p>
  <p>Built with TensorFlow & Streamlit · Cresci-17 Dataset</p>
</div>
""", unsafe_allow_html=True)