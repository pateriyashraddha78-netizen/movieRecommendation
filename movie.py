"""
Movie Recommendation System
Complete source code for a college presentation project.

Features:
- Downloads the MovieLens ml-latest-small dataset automatically.
- Builds two recommenders:
  1) Content-based recommender (TF-IDF on movie titles + genres)
  2) Collaborative-filtering (item-item cosine similarity from user ratings)
- CLI functions to get recommendations by movie title or for a given user id.
- Simple evaluation: train/test split to compute RMSE and sample recommendations.

Requirements:
- Python 3.8+
- pandas, numpy, scikit-learn, scipy

How to run:
1) Install requirements: pip install pandas numpy scikit-learn scipy
2) python Movie_Recommendation_System.py

Notes for presentation:
- Explain intuition: Content-based recommends movies similar by metadata (title/genres). Collaborative uses other users' ratings.
- Use the included example calls in main() to show outputs.

Presentation Talking Points:
1. **Introduction**: Explain the idea of recommending movies automatically.
2. **Dataset**: Introduce the MovieLens dataset and its importance.
3. **Content-Based Filtering**: Works on TF-IDF of movie titles and genres. Similarity = cosine similarity.
4. **Collaborative Filtering**: Works on user ratings. Uses item-item similarity to predict unseen movies.
5. **Demo**: Show recommendations for a movie (e.g., Toy Story) and a user (e.g., userId=1).
6. **Evaluation**: Explain how RMSE is used to measure accuracy.
7. **Conclusion**: Mention real-world applications like Netflix, Amazon, and Spotify.

"""

import os
import zipfile
import urllib.request
import io
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = "ml-latest-small"


def download_and_extract_movielens(data_url=DATA_URL, target_dir=DATA_DIR):
    """Download and extract MovieLens small dataset (if not already present).
    Returns path to extracted folder."""
    if os.path.isdir(target_dir):
        print(f"Dataset already exists at '{target_dir}'.")
        return target_dir

    print("Downloading MovieLens dataset (ml-latest-small)...")
    resp = urllib.request.urlopen(data_url)
    data = resp.read()
    print("Download complete — extracting...")
    z = zipfile.ZipFile(io.BytesIO(data))
    z.extractall()
    print("Extraction complete.")
    return target_dir


class MovieRecommender:
    def __init__(self, ratings_path, movies_path):
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        # Basic preprocessing
        self.movies['genres'] = self.movies['genres'].fillna('')
        self.movies['title'] = self.movies['title'].fillna('')
        # Prepare fields
        self._build_content_matrix()
        self._build_rating_matrix()

    # ------------------ Content-based Recommender ------------------
    def _build_content_matrix(self):
        # Combine title + genres for better similarity
        docs = (self.movies['title'] + ' ' + self.movies['genres']).values
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf.fit_transform(docs)
        self.title_to_idx = {title.lower(): idx for idx, title in enumerate(self.movies['title'].values)}

    def recommend_by_genre(self, genre, top_n=10):
        """Return top_n movies matching the selected genre."""
        genre = genre.strip()
        filtered = self.movies[self.movies['genres'].str.contains(genre, case=False, na=False)]
        return filtered.head(top_n)[['movieId', 'title', 'genres']]

    def recommend_by_movie(self, movie_title, top_n=10):
        """Return top_n similar movies to the provided movie_title using cosine similarity on TF-IDF."""
        title_key = movie_title.strip().lower()
        if title_key not in self.title_to_idx:
            matches = [t for t in self.title_to_idx.keys() if title_key in t]
            if matches:
                idx = self.title_to_idx[matches[0]]
            else:
                raise ValueError(f"Movie title '{movie_title}' not found in dataset.")
        else:
            idx = self.title_to_idx[title_key]

        cosine_similarities = linear_kernel(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()
        related_indices = cosine_similarities.argsort()[::-1]
        related_indices = [i for i in related_indices if i != idx]
        top_indices = related_indices[:top_n]
        return self.movies.iloc[top_indices][['movieId', 'title', 'genres']]

    # ------------------ Collaborative Filtering (Item-Item) ------------------
    def _build_rating_matrix(self):
        ratings_pivot = self.ratings.pivot_table(index='userId', columns='movieId', values='rating')
        self.ratings_pivot = ratings_pivot
        self.ratings_filled = ratings_pivot.fillna(0)
        self.movieid_to_col = {movieId: idx for idx, movieId in enumerate(self.ratings_filled.columns)}
        print("Computing item-item similarity (may take a few seconds)...")
        self.item_sim_matrix = cosine_similarity(self.ratings_filled.T)
        print("Item-item similarity computed.")

    def recommend_for_user(self, user_id, top_n=10):
        """Generate top-N recommendations for a given user based on item-item collaborative filtering."""
        if user_id not in self.ratings_pivot.index:
            raise ValueError(f"User id {user_id} not found in ratings.")

        user_ratings = self.ratings_pivot.loc[user_id]
        rated_items = user_ratings.dropna().index.tolist()
        if not rated_items:
            raise ValueError(f"User {user_id} has no ratings.")

        scores = np.zeros(self.item_sim_matrix.shape[0])
        for mid in rated_items:
            col_idx = self.movieid_to_col[mid]
            user_rating = user_ratings[mid]
            scores += user_rating * self.item_sim_matrix[col_idx]

        for mid in rated_items:
            scores[self.movieid_to_col[mid]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_n]
        top_movie_ids = [self.ratings_filled.columns[i] for i in top_indices]
        return self.movies[self.movies['movieId'].isin(top_movie_ids)][['movieId', 'title', 'genres']]

    # ------------------ Simple Evaluation ------------------
    def simple_train_test_eval(self, test_size=0.2, random_state=42):
        train, test = train_test_split(self.ratings, test_size=test_size, random_state=random_state)
        train_pivot = train.pivot_table(index='userId', columns='movieId', values='rating')
        train_filled = train_pivot.fillna(0)
        sim = cosine_similarity(train_filled.T)

        y_true, y_pred = [], []
        movieids = train_filled.columns
        movieid_to_col = {m: i for i, m in enumerate(movieids)}

        for _, row in test.iterrows():
            user = row['userId']
            mid = row['movieId']
            true_rating = row['rating']

            if user not in train_pivot.index:
                continue
            user_vec = train_pivot.loc[user].fillna(0)
            if mid not in movieid_to_col:
                continue
            target_col = movieid_to_col[mid]
            sims = sim[target_col]
            rated_mask = user_vec.values != 0
            if not rated_mask.any():
                continue
            numer = np.dot(sims, user_vec.fillna(0).values)
            denom = np.sum(np.abs(sims[rated_mask]))
            if denom == 0:
                continue
            pred = numer / denom
            y_true.append(true_rating)
            y_pred.append(pred)

        if not y_true:
            print("Not enough overlap to evaluate.")
            return None
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        print(f"Item-Item CF RMSE on test split: {rmse:.4f}")
        return rmse


# ------------------ Utility / Demo ------------------

def main():
    data_dir = download_and_extract_movielens()
    movies_path = os.path.join(data_dir, 'movies.csv')
    ratings_path = os.path.join(data_dir, 'ratings.csv')

    recommender = MovieRecommender(ratings_path=ratings_path, movies_path=movies_path)

    print('\n=== Content-based recommendations for: "Toy Story (1995)" ===')
    recs = recommender.recommend_by_movie('Toy Story (1995)', top_n=8)
    print(recs.to_string(index=False))

    print('\n=== Content-based recommendations for: "Forrest Gump (1994)" ===')
    recs = recommender.recommend_by_movie('Forrest Gump (1994)', top_n=8)
    print(recs.to_string(index=False))

    print('\n=== Collaborative recommendations for user id 1 ===')
    recs = recommender.recommend_for_user(user_id=1, top_n=8)
    print(recs.to_string(index=False))

    print('\n=== Quick evaluation (may take longer) ===')
    recommender.simple_train_test_eval(test_size=0.2)


if __name__ == '__main__':
    main()
