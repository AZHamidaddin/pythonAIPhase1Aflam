# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from collections import defaultdict

# Load data from CSV files
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
tags = pd.read_csv("tags.csv")

# Fill missing genre values with an empty string
movies["genres"] = movies["genres"].fillna("")

# Initialize TF-IDF Vectorizer to convert genres into numerical format
tfidf = TfidfVectorizer(stop_words="english")

# Apply TF-IDF transformation to the genres column
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Compute cosine similarity between movies based on genre
similarity_matrix = cosine_similarity(tfidf_matrix)


def get_content_based_recommendations(movie_indices, top_n=5):
    """Given a list of movie indices, find similar movies."""
    # Compute the average similarity scores for selected movies
    scores = np.mean(similarity_matrix[movie_indices], axis=0)

    # Rank movies based on similarity scores in descending order
    ranked_indices = np.argsort(scores)[::-1]

    # Return the top N recommendations excluding selected movies
    return [(movies.iloc[i]["title"], scores[i]) for i in ranked_indices if i not in movie_indices][:top_n]


# Initialize Surprise library reader for rating scale
reader = Reader(rating_scale=(0.5, 5.0))

# Load rating dataset into Surprise format
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

# Split dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize Singular Value Decomposition (SVD) model for Collaborative Filtering
model = SVD()

# Train the SVD model on training data
model.fit(trainset)

# Save the trained SVD model
with open("svd_model.pkl", "wb") as file:
    pickle.dump(model, file)


def get_collaborative_recommendations(user_id, top_n=5):
    """Generate movie recommendations for a user based on Collaborative Filtering."""
    # Identify movies that the user hasn't rated
    movie_ids = set(movies["movieId"]) - set(ratings[ratings["userId"] == user_id]["movieId"])

    # Predict ratings for unseen movies using the trained SVD model
    predictions = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in movie_ids]

    # Sort movies by predicted rating in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Return the top N recommended movies with predicted ratings
    return [(movies[movies["movieId"] == movie_id]["title"].values[0], rating) for movie_id, rating in
            predictions[:top_n]]


def interactive_recommendation():
    """Interactive console-based movie recommender system."""
    print("Welcome to the AI Movie Recommender!")

    # Ask for user's name
    user_name = input("Enter your name: ")
    print(f"Hello {user_name}, please select movies you like from the list below:")

    # Display movie list with numbering
    for i, row in movies.iterrows():
        print(f"{i + 1}. {row['title']}")

    # Initialize a list to store user-selected movies
    selected_indices = []

    while True:
        # Ask user for movie selection
        choice = input("Enter movie numbers (or -1 to finish): ")
        if choice == "-1":
            break
        try:
            # Convert user input to an index and add to the list
            selected_indices.append(int(choice) - 1)
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # If no movies are selected, prompt user to try again
    if not selected_indices:
        print("You didn't select any movies. Try again!")
        return

    print("Processing recommendations...")

    # Generate recommendations based on content similarity
    content_recs = get_content_based_recommendations(selected_indices)

    # Generate recommendations using collaborative filtering (using user 1 as a sample)
    collab_recs = get_collaborative_recommendations(1)

    print("\nHere are your AI-powered movie recommendations:")

    # Display content-based recommendations
    print("(Based on genres you like)")
    for movie, score in content_recs:
        print(f"- {movie} (similarity: {score:.2f})")

    # Display collaborative filtering recommendations
    print("\n(Based on user ratings similar to yours)")
    for movie, rating in collab_recs:
        print(f"- {movie} (predicted rating: {rating:.2f})")

    print("\nEnjoy your movies!")


# Run the recommendation system
if __name__ == "__main__":
    interactive_recommendation()

