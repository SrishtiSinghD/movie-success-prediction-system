import os
import pandas as pd
import google.generativeai as genai

# =========================================================
# GEMINI CONFIG
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# =========================================================
# LOAD DATASET
# =========================================================

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "movies_cleaned.xlsx")

df = None

try:
    df = pd.read_excel(DATA_PATH)

    if "name" in df.columns:
        df["name_lower"] = df["name"].astype(str).str.lower()

except Exception:
    df = None

# =========================================================
# MAIN FUNCTION
# =========================================================

def get_movie_summary(movie_name, context=None):

    movie_name_lower = movie_name.lower()

    prompt = ""

    # =====================================================
    # CASE 1 → MOVIE FOUND IN DATASET
    # =====================================================

    if df is not None and movie_name_lower in df["name_lower"].values:

        movie_data = df[df["name_lower"] == movie_name_lower].iloc[0]

        genre = movie_data.get("genre", "Unknown")
        year = movie_data.get("year", "Unknown")
        director = movie_data.get("director", "Unknown")
        writer = movie_data.get("writer", "Unknown")
        star = movie_data.get("star", "Unknown")
        rating = movie_data.get("rating", "Unknown")
        score = movie_data.get("score", "Unknown")
        votes = movie_data.get("votes", "Unknown")
        country = movie_data.get("country", "Unknown")
        runtime = movie_data.get("runtime", "Unknown")
        company = movie_data.get("company", "Unknown")
        budget = movie_data.get("budget", "Unknown")
        gross = movie_data.get("gross", "Unknown")

        prompt = f"""
        Generate a professional and engaging movie summary
        in around 80-100 words.

        Movie Title: {movie_name}

        Genre: {genre}
        Release Year: {year}

        Director: {director}
        Writer: {writer}

        Main Star: {star}

        Rating: {rating}
        IMDb Score: {score}
        Votes: {votes}

        Country: {country}
        Runtime: {runtime} minutes

        Production Company: {company}

        Budget: ${budget}
        Gross Collection: ${gross}

        Create a cinematic, attractive, and polished summary.
        """

    # =====================================================
    # CASE 2 → MOVIE NOT FOUND
    # =====================================================

    else:

        prompt = f"""
        Generate a professional and engaging movie summary
        in around 80-100 words.

        Movie Title: {movie_name}

        Create a cinematic and attractive summary.
        """

    # =====================================================
    # GENERATE RESPONSE
    # =====================================================

    try:
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Gemini Error: {str(e)}"