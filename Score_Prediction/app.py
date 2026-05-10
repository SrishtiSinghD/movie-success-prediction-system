#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================================
# LOAD MODEL FILES (same folder as app.py)
# ======================================
model = joblib.load("movie_score_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ======================================
# SIMPLE MAPPINGS FOR USER-FRIENDLY INPUTS
# ======================================
genre_map = {
    "Action": 0,
    "Comedy": 1,
    "Drama": 2,
    "Horror": 3,
    "Romance": 4,
    "Sci-Fi": 5,
    "Thriller": 6,
    "Adventure": 7,
    "Animation": 8,
    "Crime": 9,
}

rating_map = {
    "G": 0,
    "PG": 1,
    "PG-13": 2,
    "R": 3,
    "NC-17": 4,
    "Unrated": 5,
}

country_map = {
    "USA": 0,
    "UK": 1,
    "India": 2,
    "Canada": 3,
    "France": 4,
    "Germany": 5,
    "Japan": 6,
    "Other": 7,
}


def text_to_number(text: str) -> int:
    """Convert any text to a stable numeric value."""
    if not text.strip():
        return 0
    return abs(hash(text.lower())) % 1000


# ======================================
# PAGE SETTINGS
# ======================================
st.set_page_config(
    page_title="IMDb Movie Score Prediction",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 IMDb Movie Score Prediction")
st.write("Enter movie details to predict the IMDb score and whether the movie is likely to be a hit.")

# ======================================
# USER INPUTS
# ======================================
col1, col2 = st.columns(2)

with col1:
    genre_name = st.selectbox("🎭 Genre", list(genre_map.keys()))
    rating_name = st.selectbox("🔞 Content Rating", list(rating_map.keys()))
    year = st.number_input("📅 Release Year", min_value=1980, max_value=2035, value=2024)
    runtime = st.number_input("⏱️ Runtime (minutes)", min_value=60, max_value=300, value=120)
    release_month = st.slider("🗓️ Release Month", 1, 12, 6)

with col2:
    director_name = st.text_input("🎬 Director", "Christopher Nolan")
    writer_name = st.text_input("✍️ Writer", "Jonathan Nolan")
    star_name = st.text_input("⭐ Lead Actor", "Leonardo DiCaprio")
    country_name = st.selectbox("🌍 Country", list(country_map.keys()))
    company_name = st.text_input("🏢 Production Company", "Warner Bros.")

st.subheader("💰 Production Details")

col3, col4 = st.columns(2)
with col3:
    votes = st.number_input("🗳️ Number of Votes", min_value=0, value=50000, step=1000)
with col4:
    budget = st.number_input("💵 Budget (USD)", min_value=0, value=10000000, step=1000000)

# ======================================
# ENCODE USER INPUTS
# ======================================
genre = genre_map[genre_name]
rating = rating_map[rating_name]
country = country_map[country_name]

director = text_to_number(director_name)
writer = text_to_number(writer_name)
star = text_to_number(star_name)
company = text_to_number(company_name)

# ======================================
# FEATURE ENGINEERING
# ======================================
decade = (year // 10) * 10
log_votes = np.log1p(votes)
log_budget = np.log1p(budget)
budget_available = 1 if budget > 0 else 0

# ======================================
# BUILD INPUT DATAFRAME
# ======================================
input_data = pd.DataFrame([
    {
        "genre": genre,
        "rating": rating,
        "year": year,
        "runtime": runtime,
        "director": director,
        "writer": writer,
        "star": star,
        "country": country,
        "company": company,
        "release_month": release_month,
        "decade": decade,
        "log_votes": log_votes,
        "log_budget": log_budget,
        "budget_available": budget_available,
    }
])

# Ensure exact column order expected by the model
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# ======================================
# PREDICT BUTTON
# ======================================
if st.button("🔮 Predict IMDb Rating", use_container_width=True):
    predicted_score = round(float(model.predict(input_data)[0]), 2)

    # Main Result
    st.success(f"⭐ Predicted IMDb Rating: {predicted_score}/10")

    # Hit / Flop
    if predicted_score >= 7.0:
        st.balloons()
        st.subheader("🎉 HIT MOVIE")
        st.write("This movie is likely to receive strong audience ratings.")
    else:
        st.subheader("❌ FLOP / AVERAGE MOVIE")
        st.write("This movie may receive mixed or below-average audience ratings.")

    # Quality Category
    if predicted_score >= 8.0:
        category = "🏆 Excellent Movie"
        progress = 95
    elif predicted_score >= 7.0:
        category = "🎯 Very Good Movie"
        progress = 80
    elif predicted_score >= 6.0:
        category = "👍 Good Movie"
        progress = 65
    elif predicted_score >= 5.0:
        category = "😐 Average Movie"
        progress = 45
    else:
        category = "🚫 Poorly Rated Movie"
        progress = 25

    st.write(category)
    st.progress(progress)
    st.metric("Predicted IMDb Score", f"{predicted_score}/10")

    # Feature Importance
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "Feature": feature_columns,
                "Importance": model.feature_importances_,
            }
        )

        importance_df = (
            importance_df.sort_values("Importance", ascending=False)
            .head(10)
            .set_index("Feature")
        )

        st.subheader("🎯 Top Influencing Factors")
        st.bar_chart(importance_df)

