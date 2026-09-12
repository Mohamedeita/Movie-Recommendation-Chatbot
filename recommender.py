import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. Load Data
# ==========================================

movies = pd.read_csv("movies.csv")
tags = pd.read_csv("tags.csv")


# ==========================================
# 2. Arabic Movie Names
# ==========================================

ARABIC_MOVIE_NAMES = {
    "باتمان": "batman",
    "الرجل الوطواط": "batman",
    "الجوكر": "joker",
    "جوكر": "joker",
    "تايتانيك": "titanic",
    "ماتريكس": "matrix",
    "ذا ماتريكس": "matrix",
    "العراب": "godfather",
    "سبايدرمان": "spider-man",
    "سبايدر مان": "spider-man",
    "سوبرمان": "superman",
    "هاري بوتر": "harry potter",
    "جيمس بوند": "james bond",
}


def translate_movie_name(message):
    """
    Convert known Arabic movie names
    into English search keywords.
    """

    message = message.lower()

    for arabic_name, english_name in ARABIC_MOVIE_NAMES.items():

        if arabic_name in message:
            return english_name

    return None


# ==========================================
# 3. Clean Genres
# ==========================================

movies["genres_clean"] = movies["genres"].replace(
    "(no genres listed)",
    ""
)


# ==========================================
# 4. Prepare Tags
# ==========================================

movie_tags = (
    tags.groupby("movieId")["tag"]
    .apply(lambda x: " ".join(x))
    .reset_index()
)


# ==========================================
# 5. Merge Movies + Tags
# ==========================================

movies_with_tags = movies.merge(
    movie_tags,
    on="movieId",
    how="left"
)

movies_with_tags["tag"] = (
    movies_with_tags["tag"]
    .fillna("")
)


# ==========================================
# 6. Combined Features
# ==========================================

movies_with_tags["combined_features"] = (
    movies_with_tags["genres_clean"]
    + " "
    + movies_with_tags["tag"]
)


# ==========================================
# 7. TF-IDF
# ==========================================

tfidf = TfidfVectorizer()

combined_matrix = tfidf.fit_transform(
    movies_with_tags["combined_features"]
)


# ==========================================
# 8. Cosine Similarity
# ==========================================

combined_similarity = cosine_similarity(
    combined_matrix
)


# ==========================================
# 9. Find Movie From Message
# ==========================================

def find_movie_from_message(message):

    message = message.lower()

    # --------------------------------------
    # Try Arabic movie names
    # --------------------------------------

    translated_name = translate_movie_name(
        message
    )

    if translated_name is not None:

        for title in movies["title"]:

            clean_title = (
                title.lower()
                .split(" (")[0]
            )

            if translated_name in clean_title:

                return title

    # --------------------------------------
    # Try English movie names
    # --------------------------------------

    matches = []

    for title in movies["title"]:

        clean_title = (
            title.lower()
            .split(" (")[0]
        )

        if clean_title in message:

            matches.append(
                (title, len(clean_title))
            )

    if matches:

        return max(
            matches,
            key=lambda x: x[1]
        )[0]

    return None


# ==========================================
# 10. Extract Number
# ==========================================

def extract_number(message):

    # English digits
    numbers = re.findall(
        r"\b(?:10|[1-9])\b",
        message
    )

    if numbers:

        number = int(numbers[0])

        if 1 <= number <= 10:

            return number

    # English number words
    number_words = {

        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10

    }

    for word, value in number_words.items():

        if re.search(
            rf"\b{word}\b",
            message
        ):

            return value

    return None


# ==========================================
# 11. Extract Genre
# ==========================================

ENGLISH_GENRES = [

    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "horror",
    "mystery",
    "romance",
    "sci-fi",
    "thriller",
    "war",
    "western"

]


def extract_genre(message):

    message = message.lower()

    for genre in ENGLISH_GENRES:

        if genre in message:

            return genre

    return None


# ==========================================
# 12. Recommend Similar Movies
# ==========================================

def recommend_similar_movies(
    movie_title,
    n=5,
    genre=None
):

    # Find movie index
    movie_index = movies_with_tags[
        movies_with_tags["title"] == movie_title
    ].index[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(
            combined_similarity[movie_index]
        )
    )

    # Sort from highest similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    # Skip index 0 because it is the same movie
    for index, score in similarity_scores[1:]:

        movie = movies_with_tags.iloc[index]

        # Optional genre filter
        if genre is not None:

            movie_genres = (
                movie["genres"]
                .lower()
            )

            if genre not in movie_genres:

                continue

        recommendations.append({

            "title": movie["title"],

            "similarity": round(
                float(score),
                3
            ),

            "genres": movie["genres"]

        })

        if len(recommendations) >= n:

            break

    return recommendations


# ==========================================
# 13. Chatbot Response
# ==========================================

def chatbot_response(
    message,
    n=5,
    last_movie=None
):

    # --------------------------------------
    # Find movie in current message
    # --------------------------------------

    movie = find_movie_from_message(
        message
    )

    # --------------------------------------
    # Use previous movie if needed
    # --------------------------------------

    if movie is None:

        movie = last_movie

    # --------------------------------------
    # No movie found
    # --------------------------------------

    if movie is None:

        return {

            "movie": None,

            "recommendations": [],

            "number": n,

            "genre": None,

            "message": (
                "I couldn't identify the movie. "
                "Please mention the movie name."
            )

        }

    # --------------------------------------
    # Extract requested number
    # --------------------------------------

    extracted_number = extract_number(
        message
    )

    if extracted_number is not None:

        n = extracted_number

    # --------------------------------------
    # Extract genre
    # --------------------------------------

    genre = extract_genre(
        message
    )

    # --------------------------------------
    # Get recommendations
    # --------------------------------------

    recommendations = (
        recommend_similar_movies(
            movie_title=movie,
            n=n,
            genre=genre
        )
    )

    # --------------------------------------
    # If genre gives no results,
    # return normal recommendations
    # --------------------------------------

    if (
        len(recommendations) == 0
        and genre is not None
    ):

        recommendations = (
            recommend_similar_movies(
                movie_title=movie,
                n=n,
                genre=None
            )
        )

    # --------------------------------------
    # Create response
    # --------------------------------------

    if genre is not None:

        response_message = (

            f"If you liked **{movie}**, "
            f"you might like these "
            f"**{genre}** movies:"

        )

    else:

        response_message = (

            f"If you liked **{movie}**, "
            "you might also like these movies:"

        )

    # --------------------------------------
    # Return result
    # --------------------------------------

    return {

        "movie": movie,

        "recommendations": recommendations,

        "number": n,

        "genre": genre,

        "message": response_message

    }