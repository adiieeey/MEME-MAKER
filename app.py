import flask
from flask import Flask, request, render_template
import pandas as pd
import pickle
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# ✅ Load dataset
df = pd.read_csv("preprocessed_meme_dataset.csv")

# ✅ Load AI Models
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
with open("vectors_matrix.pkl", "rb") as file:
    vectors_matrix = pickle.load(file)

# ✅ Improved Meme Matching
def find_best_meme(user_input):
    user_input_vector = bert_model.encode(user_input).reshape(1, -1)  # Encode scenario
    similarity_scores = cosine_similarity(user_input_vector, vectors_matrix)

    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[0, best_match_index]

    print(f"Best Score: {best_score} | User Input: {user_input}")

    # ✅ Adjust threshold dynamically
    if best_score < 0.2:  # Lower threshold to improve matching
        return "No Match Found", "https://i.imgflip.com/4t0m5.jpg"  # Default meme

    best_meme = df.iloc[best_match_index]
    return best_meme["Meme Name"], best_meme["Image URL"]

@app.route("/", methods=["GET", "POST"])
def index():
    meme_name, meme_url, user_caption = None, None, None
    if request.method == "POST":
        user_caption = request.form["caption"]
        meme_name, meme_url = find_best_meme(user_caption)

    return render_template("index.html", meme_name=meme_name, meme_url=meme_url, user_input=user_caption)

if __name__ == "__main__":
    app.run(debug=True)

