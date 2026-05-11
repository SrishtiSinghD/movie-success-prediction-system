#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
# USER-FRIENDLY DROPDOWN OPTIONS
# ======================================
# Note: The numeric values are simple placeholder encodings.
# They do not need to match exact label-encoder values for a class demo.

GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"
]

RATINGS = ["G", "PG", "PG-13", "R", "NC-17"]

COUNTRIES = [
    "USA", "India", "UK", "Canada", "France",
    "Germany", "Japan", "South Korea", "Australia"
]

DIRECTORS = [
    "Christopher Nolan", "Steven Spielberg", "James Cameron",
    "Rajkumar Hirani", "S. S. Rajamouli", "Quentin Tarantino",
    "Greta Gerwig", "Martin Scorsese", "David Fincher"
]

WRITERS = [
    "Jonathan Nolan", "Aaron Sorkin", "Vijayendra Prasad",
    "Charlie Kaufman", "Greta Gerwig", "Akiva Goldsman"
]

STARS = [
    "Leonardo DiCaprio", "Tom Cruise", "Robert Downey Jr.",
    "Shah Rukh Khan", "Prabhas", "Scarlett Johansson",
    "Emma Stone", "Christian Bale"
]

COMPANIES = [
    "Warner Bros.", "Universal Pictures", "Disney",
    "Paramount Pictures", "Netflix", "Marvel Studios",
    "Dharma Productions", "Arka Media Works"
]

YEARS = list(range(2000, 2027))
RUNTIMES = [90, 100, 110, 120, 130, 140, 150, 180]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
VOTES_OPTIONS = [5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
BUDGET_OPTIONS = [
    1000000, 5000000, 10000000, 25000000,
    50000000, 100000000, 200000000, 300000000
]


def encode_from_list(value, options):
    """Encode a selected dropdown value as its index."""
    return options.index(value)


# ======================================
# PAGE SETTINGS
# ======================================
st.set_page_config(
    page_title="IMDb Movie Score Prediction",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 IMDb Movie Score Prediction")
st.write(
    "Select movie details to predict the IMDb score and determine whether the movie is likely to be a hit."
)

# ======================================
# USER INPUTS (ALL DROPDOWNS)
# ======================================
col1, col2 = st.columns(2)

with col1:
    genre_name = st.selectbox("🎭 Genre", GENRES, index=0)
    rating_name = st.selectbox("🔞 Content Rating", RATINGS, index=2)  # PG-13
    year = st.selectbox("📅 Release Year", YEARS, index=len(YEARS) - 3)
    runtime = st.selectbox("⏱️ Runtime (minutes)", RUNTIMES, index=3)  # 120
    month_name = st.selectbox("🗓️ Release Month", MONTHS, index=5)  # June

with col2:
    director_name = st.selectbox("🎬 Director", DIRECTORS, index=0)
    writer_name = st.selectbox("✍️ Writer", WRITERS, index=0)
    star_name = st.selectbox("⭐ Lead Actor", STARS, index=0)
    country_name = st.selectbox("🌍 Country", COUNTRIES, index=0)
    company_name = st.selectbox("🏢 Production Company", COMPANIES, index=0)

st.subheader("💰 Production Details")

col3, col4 = st.columns(2)
with col3:
    votes = st.selectbox("🗳️ Expected Number of IMDb Votes", VOTES_OPTIONS, index=4)
with col4:
    budget = st.selectbox("💵 Budget (USD)", BUDGET_OPTIONS, index=4)

# ======================================
# ENCODE USER INPUTS
# ======================================
genre = encode_from_list(genre_name, GENRES)
rating = encode_from_list(rating_name, RATINGS)
country = encode_from_list(country_name, COUNTRIES)
director = encode_from_list(director_name, DIRECTORS)
writer = encode_from_list(writer_name, WRITERS)
star = encode_from_list(star_name, STARS)
company = encode_from_list(company_name, COMPANIES)
release_month = MONTHS.index(month_name) + 1

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

    # Clamp to realistic IMDb range
    predicted_score = max(1.0, min(10.0, predicted_score))

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
                "Importance": model.feature_importances_,  # type: ignore[attr-defined]
            }
        )

        importance_df = (
            importance_df.sort_values("Importance", ascending=False)
            .head(10)
            .set_index("Feature")
        )

        st.subheader("🎯 Top Influencing Factors")
        st.bar_chart(importance_df)


# In[ ]:





# In[ ]:




