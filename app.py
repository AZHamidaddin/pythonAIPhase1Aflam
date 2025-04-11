from flask import Flask, render_template, request, redirect, url_for
from main import movies, get_content_based_recommendations, get_collaborative_recommendations

app = Flask(__name__)
selected_indices = []

@app.route("/")
def index():
    return redirect(url_for('select_movies'))

@app.route("/select", methods=["GET", "POST"])
def select_movies():
    global selected_indices
    genres = sorted(set(g for gs in movies['genres'] for g in gs.split('|') if g))
    if request.method == "POST":
        form_data = request.form.get("movie_indices", "")
        selected_indices = list(map(int, form_data.split(","))) if form_data else []
        return redirect(url_for("recommend"))
    return render_template("movies.html", movies=movies.iterrows(), genres=genres)

@app.route("/recommend")
def recommend():
    content_recs = get_content_based_recommendations(selected_indices)
    collab_recs = get_collaborative_recommendations(1)
    return render_template("recommendations.html", content=content_recs, collab=collab_recs)

if __name__ == "__main__":
    app.run(debug=True)
