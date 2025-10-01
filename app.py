from flask import Flask, render_template, request
import os
from movie import MovieRecommender, DATA_DIR, download_and_extract_movielens

app = Flask(__name__)

# Load recommender (reuse logic, do not modify movie.py)
data_dir = download_and_extract_movielens()
movies_path = os.path.join(data_dir, 'movies.csv')
ratings_path = os.path.join(data_dir, 'ratings.csv')
recommender = MovieRecommender(ratings_path=ratings_path, movies_path=movies_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    recs = None
    error = None
    mode = None
    genre = None
    if request.method == 'POST':
        mode = request.form.get('mode')
        if mode == 'movie':
            title = request.form.get('movie_title')
            try:
                recs = recommender.recommend_by_movie(title, top_n=8)
            except Exception as e:
                error = str(e)
        elif mode == 'user':
            user_id = request.form.get('user_id')
            try:
                recs = recommender.recommend_for_user(int(user_id), top_n=8)
            except Exception as e:
                error = str(e)
        elif mode == 'genre':
            genre = request.form.get('genre')
            try:
                recs = recommender.recommend_by_genre(genre, top_n=8)
            except Exception as e:
                error = str(e)
    return render_template('index.html', recs=recs, error=error, mode=mode, genre=genre)

@app.route('/genres')
def genres():
    # For future API use, not used in template
    genres_list = [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    return {"genres": genres_list}
if __name__ == '__main__':
    app.run(debug=True)