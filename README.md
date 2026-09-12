# 🎬 MovieMind - Movie Recommendation Chatbot

MovieMind is a movie recommendation chatbot built using Machine Learning.

The project recommends movies based on the movie the user likes by comparing movie features using TF-IDF and Cosine Similarity.

## Project Features

* Movie recommendations using Machine Learning
* Content-Based Recommendation System
* Uses movie genres and tags
* TF-IDF for feature extraction
* Cosine Similarity to find similar movies
* Simple chatbot interface using Streamlit
* Supports movie names in English
* Supports some Arabic movie-name inputs
* Users can request a specific number of recommendations
* Optional genre-based recommendations

## Dataset

The project uses the MovieLens dataset.

The main files used are:

* `movies.csv` - movie titles and genres
* `tags.csv` - user-generated movie tags

## How It Works

The recommendation system follows these steps:

```text
Movie Input
     ↓
Find Movie
     ↓
Genres + Tags
     ↓
TF-IDF
     ↓
Movie Feature Vectors
     ↓
Cosine Similarity
     ↓
Similar Movies
     ↓
Recommendations
```

### 1. Data Preparation

Movie genres and tags are combined into one feature column.

### 2. TF-IDF

TF-IDF converts the movie features into numerical vectors.

### 3. Cosine Similarity

Cosine Similarity measures how similar two movies are based on their features.

### 4. Recommendation

The system selects the movies with the highest similarity scores and recommends them to the user.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* TF-IDF
* Cosine Similarity

## Project Structure

```text
Movie-Recommendation-Chatbot/
│
├── Forntend.py
├── recommender.py
├── movies.csv
├── tags.csv
├── requirements.txt
└── ChatBot_Movie_Recommendation.ipynb
```

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Mohamedeita/Movie-Recommendation-Chatbot.git
```

### 2. Open the project folder

```bash
cd Movie-Recommendation-Chatbot
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run Forntend.py
```

The application will open in your browser.

## Example

You can enter:

```text
I watched Jaws and liked it
```

The chatbot will analyze the movie and recommend similar movies.

You can also ask:

```text
Give me 10 movies like Batman
```

Or:

```text
I want action movies like Batman
```

## Author

**Mohamed Eita**

LinkedIn:
https://www.linkedin.com/in/mohamed-eita-581187371
