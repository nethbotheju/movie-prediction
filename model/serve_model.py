import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify

# -------------------------
# 1. LOAD DATA AND INDEX
# -------------------------
def load_data_and_index(metadata_path='movie_metadata.csv', index_path='movie_index.faiss'):
    """
    Load metadata and FAISS index.
    """
    df = pd.read_csv(metadata_path)
    index = faiss.read_index(index_path)
    return df, index

# -------------------------
# 2. RETRIEVAL FUNCTION
# -------------------------
def get_top_movie(query_summary, df, index, model, top_k=1):
    """
    Retrieve top movie based on query summary.
    """
    query_emb = model.encode([query_summary]).astype('float32')
    
    # Search
    distances, indices = index.search(query_emb, top_k)
    
    # Get results
    top_movies = df.iloc[indices[0]].to_dict('records')
    return top_movies

# -------------------------
# 3. FLASK API
# -------------------------
app = Flask(__name__)

# Load data and index at startup
df, index = load_data_and_index()
model = SentenceTransformer('fine_tuned_model')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    query = data['summary']
    
    # Get top movie (adjust top_k if needed)
    top_movie = get_top_movie(query, df, index, model, top_k=1)[0]
    
    return jsonify({
        'movie_name': top_movie['movie name'],
        'plot': top_movie['plot'],
        'year': top_movie['year'],
        'genres': top_movie['genres']
    })

if __name__ == '__main__':
    app.run(port=5000)