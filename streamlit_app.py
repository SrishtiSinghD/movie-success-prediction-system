"""
streamlit_app.py
================
Streamlit UI for the Movie ROI Prediction Pipeline.

Features
--------
  Tab 1 — Predict a Movie
    • Form with all movie attributes (budget optional)
    • Predicts gross revenue + ROI class
    • Shows pipeline used (A or B) + imputed budget if applicable
    • Probability bar chart per ROI class

  Tab 2 — Movie Summarizer  (Gemini — placeholder)
    • Enter a movie name → AI-generated summary
    • Placeholder until gemini_service.py is implemented

Run
---
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from config import ROI_LABELS
from models import RoiClassifier
from train import predict_single_movie
from gemini_service import get_movie_summary

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "🎬 Movie ROI Predictor",
    page_icon  = "🎬",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────

for key, default in [
    ("clf",       None),
    ("metrics",   {}),
    ("charts",    {}),
    ("trained",   False),
    ("comparison", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Model Controls")

    # Load pre-trained model
    st.subheader("Upload Trained Model")

    load_file = st.file_uploader(
        "Upload roi_classifier.pkl",
        type=["pkl"]
    )

    if load_file is not None:
        try:
            clf_loaded = pickle.load(load_file)

            st.session_state["clf"] = clf_loaded
            st.session_state["trained"] = True

            if st.session_state.get("metrics") is None:
                st.session_state["metrics"] = {}

            st.success("✅ Model loaded successfully!")

        except Exception as e:
            st.error(f"Failed to load model: {e}")

    st.markdown("---")

    st.caption(
        "**ROI Bins (Cocuzzo & Wu 2013)**\n\n"
        "- [0, 0.5)  → 💀 flop\n"
        "- [0.5, 1)  → 📉 below_avg\n"
        "- [1, 2)    → ✅ hit\n"
        "- [2, ∞)    → 🚀 blockbuster"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main title
# ─────────────────────────────────────────────────────────────────────────────

st.title("🎬 Movie Box-Office ROI Predictor")
st.markdown(
    "Two-stage XGBoost pipeline implementing the "
    "[Cocuzzo & Wu (2013)](https://scholar.google.com) *Hit or Flop* methodology. "
    "**Pipeline A** when budget is known; **Pipeline B** auto-imputes missing budgets."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_predict, tab_gemini = st.tabs([
    "🎯 Predict a Movie",
    "✨ Movie Summarizer (AI)",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Predict a Single Movie
# ═══════════════════════════════════════════════════════════════════════════════

with tab_predict:
    st.subheader("🎯 Predict ROI Class & Gross Revenue")

    if not st.session_state["trained"]:
        st.warning("Train the model first (Tab 1) or load a saved .pkl from the sidebar.")
    else:
        st.info(
            "Fill in the movie details below. "
            "Leave **Budget** at 0 if unknown — Pipeline B will auto-impute it."
        )

        with st.form("predict_form", clear_on_submit=False):
            st.markdown("##### 🎬 Basic Info")
            r1c1, r1c2, r1c3 = st.columns(3)
            movie_name  = r1c1.text_input("Movie Name", "My New Film")
            genre       = r1c2.selectbox("Genre", [
                "Action", "Comedy", "Drama", "Horror", "Animation",
                "Biography", "Crime", "Adventure", "Romance", "Thriller",
                "Sci-Fi", "Fantasy", "Family", "Mystery", "Documentary",
                "Sport", "Music", "History", "Western",
            ])
            mpaa_rating = r1c3.selectbox("MPAA Rating", ["PG", "PG-13", "R", "G", "NC-17", "NR"])

            st.markdown("##### 📅 Release Info")
            r2c1, r2c2, r2c3 = st.columns(3)
            year    = r2c1.number_input("Release Year",  min_value=1980, max_value=2035, value=2025)
            runtime = r2c2.number_input("Runtime (min)", min_value=60,   max_value=360,  value=110)
            month   = r2c3.selectbox("Release Month", list(range(1, 13)), index=5,
                                     format_func=lambda m: [
                                         "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m])

            st.markdown("##### 💰 Production & Popularity")
            r3c1, r3c2 = st.columns(2)
            budget = r3c1.number_input(
                "Budget ($) — enter 0 if unknown",
                min_value=0, value=0, step=500_000, format="%d",
            )
            votes  = r3c2.number_input(
                "Estimated IMDb Votes",
                min_value=0, value=50_000, step=5_000, format="%d",
            )

            st.markdown("##### 🎭 Creative Team")
            r4c1, r4c2, r4c3 = st.columns(3)
            director = r4c1.text_input("Director", "Unknown Director")
            star     = r4c2.text_input("Lead Star",  "Unknown Star")
            company  = r4c3.text_input("Studio / Company", "Independent")

            submitted = st.form_submit_button("🔮 Predict", type="primary")

        if submitted:
            clf = st.session_state["clf"]
            movie_dict = {
                "name"          : movie_name,
                "genre"         : genre,
                "rating"        : mpaa_rating,
                "year"          : int(year),
                "runtime"       : float(runtime),
                "release_month" : int(month),
                "release_day"   : 15,
                "votes"         : float(votes),
                "director"      : director,
                "writer"        : director,
                "star"          : star,
                "company"       : company,
                "country"       : "United States",
                "budget"        : float(budget) if budget > 0 else None,
                "gross"         : None,
                "score"         : None,
            }

            with st.spinner("Running prediction …"):
                try:
                    result = predict_single_movie(clf, movie_dict)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.exception(exc)
                    st.stop()

            roi  = result["predicted_roi_class"]
            EMOJI = {
                "flop"        : "💀",
                "below_avg"   : "📉",
                "hit"         : "✅",
                "blockbuster" : "🚀",
            }
            COLOR = {
                "flop"        : "#EF4444",
                "below_avg"   : "#F59E0B",
                "hit"         : "#10B981",
                "blockbuster" : "#3B82F6",
            }

            st.divider()
            st.markdown(
                f"<h2 style='text-align:center; color:{COLOR[roi]};'>"
                f"{EMOJI[roi]} {roi.upper().replace('_', ' ')}"
                f"</h2>",
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Gross", f"${result['predicted_gross_usd']:,.0f}")
            m2.metric("Pipeline Used",   result["pipeline_used"])
            m3.metric("Budget Imputed",  "Yes" if result["budget_imputed"] else "No")

            if result.get("imputed_budget_usd"):
                st.info(f"Imputed budget: **${result['imputed_budget_usd']:,.0f}**")

            # Probability chart
            proba = result.get("roi_probability", {})
            if proba:
                st.subheader("Confidence by ROI Class")
                fig = go.Figure(go.Bar(
                    x     = list(proba.keys()),
                    y     = [v * 100 for v in proba.values()],
                    marker_color = [COLOR.get(k, "#6B7280") for k in proba],
                    text  = [f"{v*100:.1f}%" for v in proba.values()],
                    textposition = "outside",
                ))
                fig.update_layout(
                    yaxis_title = "Confidence (%)",
                    yaxis_range = [0, 110],
                    height      = 320,
                    showlegend  = False,
                    margin      = dict(t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Movie Summarizer (Gemini Placeholder)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_gemini:

    st.subheader("✨ AI Movie Summary Generator")

    movie_name = st.text_input(
        "Enter Movie Name",
        placeholder="e.g. Inception"
    )

    if st.button("Generate Summary"):

        if movie_name.strip() == "":
            st.warning("Please enter a movie name.")

        else:

            with st.spinner("Generating AI summary..."):

                summary = get_movie_summary(movie_name)

                st.success("Summary Generated!")

                st.markdown("### 🎬 AI Generated Summary")

                st.write(summary)