import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from flask_cors import CORS


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
def get_top_movies(query_summary, df, index, model, top_k=3):
    """
    Retrieve top movies based on query summary.
    """
    query_emb = model.encode([query_summary]).astype('float32')
    
    # Search for top_k movies
    distances, indices = index.search(query_emb, top_k)
    
    # Get results with similarity scores
    top_movies = []
    for i, movie_idx in enumerate(indices[0]):
        if movie_idx < len(df) and movie_idx >= 0:  # Check if index is valid
            movie = df.iloc[movie_idx].to_dict()
            # Add similarity score (convert distance to similarity)
            similarity = 1 / (1 + distances[0][i])
            movie['similarity_score'] = float(f"{similarity:.4f}")
            top_movies.append(movie)
    
    return top_movies

# -------------------------
# 3. FLASK API
# -------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

# Load data and index at startup
df, index = load_data_and_index()
model = SentenceTransformer('fine_tuned_model')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    query = data['summary']
    
    # Get top 3 movies
    top_movies = get_top_movies(query, df, index, model, top_k=3)
    
    # Format response with all three movies
    response = {
        'predictions': [
            {
                'movie_name': movie['movie name'],
                'year': movie['year'],
                'genres': movie['genres'],
                'similarity_score': movie['similarity_score']
            } for movie in top_movies
        ]
    }
    
    return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint to verify API is running
    """
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
