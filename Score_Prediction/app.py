#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load(r"C:\Users\Sahithi\Desktop\Accenture training\movie_score_model.pkl")
feature_columns = joblib.load(r"C:\Users\Sahithi\Desktop\Accenture training\feature_columns.pkl")

# =========================
# TITLE
# =========================

st.title("🎬 IMDb Movie Score Prediction")

st.write("Enter movie details to predict IMDb score.")

# =========================
# USER INPUTS
# =========================

genre = st.number_input("Genre Encoded Value", value=0)
rating = st.number_input("Rating Encoded Value", value=0)
year = st.number_input("Year", value=2024)
runtime = st.number_input("Runtime", value=120)

director = st.number_input("Director Encoded Value", value=0)
writer = st.number_input("Writer Encoded Value", value=0)
star = st.number_input("Star Encoded Value", value=0)

country = st.number_input("Country Encoded Value", value=0)
company = st.number_input("Company Encoded Value", value=0)

release_month = st.slider("Release Month", 1, 12, 6)

votes = st.number_input("Votes", value=50000)
budget = st.number_input("Budget", value=10000000)

# =========================
# FEATURE ENGINEERING
# =========================

decade = (year // 10) * 10

log_votes = np.log1p(votes)

budget_available = 1 if budget > 0 else 0

log_budget = np.log1p(budget)

# =========================
# CREATE INPUT DATAFRAME
# =========================

input_data = pd.DataFrame([{
    'genre': genre,
    'rating': rating,
    'year': year,
    'runtime': runtime,
    'director': director,
    'writer': writer,
    'star': star,
    'country': country,
    'company': company,
    'release_month': release_month,
    'decade': decade,
    'log_votes': log_votes,
    'log_budget': log_budget,
    'budget_available': budget_available
}])

# Ensure correct column order
input_data = input_data[feature_columns]

# =========================
# PREDICTION SECTION
# =========================

if st.button("Predict IMDb Rating"):

    # Predict IMDb score
    prediction = model.predict(input_data)[0]

    imdb_rating = round(float(prediction), 2)

    # Display predicted rating
    st.success(f"⭐ Predicted IMDb Rating: {imdb_rating}/10")

    # =========================
    # MOVIE SEGMENTATION
    # =========================

    if imdb_rating >= 8.0:
        category = "🏆 Excellent Movie"
        description = "Critically acclaimed with outstanding audience reception."
        progress = 95

    elif imdb_rating >= 7.0:
        category = "🎉 Very Good Movie"
        description = "Strong ratings and positive audience feedback."
        progress = 80

    elif imdb_rating >= 6.0:
        category = "👍 Average / Good Movie"
        description = "Generally liked by viewers with decent reception."
        progress = 65

    elif imdb_rating >= 5.0:
        category = "😐 Below Average Movie"
        description = "Mixed reviews and moderate audience response."
        progress = 45

    else:
        category = "❌ Poorly Rated Movie"
        description = "Low audience satisfaction and weak ratings."
        progress = 25

    # Display category
    st.subheader(category)

    # Description
    st.write(description)

    # Progress bar
    st.progress(progress)

    # Metric display
    st.metric(
        label="Predicted IMDb Score",
        value=f"{imdb_rating}/10"
    )

    # =========================
    # FEATURE IMPORTANCE
    # =========================

    if hasattr(model, 'feature_importances_'):

        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by='Importance',
            ascending=False
        ).head(10)

        st.subheader("🎯 Top Influencing Factors")

        st.bar_chart(
            importance_df.set_index('Feature')
        )


# In[ ]:





# In[ ]:




