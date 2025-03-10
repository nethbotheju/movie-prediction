import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# -------------------------
# 1. EMBEDDING GENERATION
# -------------------------
def generate_embeddings(df, model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings using Sentence-BERT.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df['combined_text'].values, convert_to_tensor=True)
    embeddings = embeddings.cpu()
    return embeddings.numpy()

# -------------------------
# 2. FINE-TUNE THE MODEL
# -------------------------
def fine_tune_model(df, model_name='all-MiniLM-L6-v2', epochs=3):
    """
    Fine-tune the Sentence-BERT model on the dataset.
    """
    model = SentenceTransformer(model_name)
    
    # Prepare training data (triplet loss for movie disambiguation)
    train_examples = []
    for _, row in df.iterrows():
        combined_text = row['combined_text']
        title = row['movie name']
        negative_title = df.sample(1)['movie name'].iloc[0]
        train_examples.append(InputExample(texts=[combined_text, title, negative_title]))
    
    # Train the model
    train_dataloader = DataLoader(train_examples, batch_size=16, shuffle=True)
    train_loss = losses.TripletLoss(model)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path='fine_tuned_model'
    )
    
    return SentenceTransformer('fine_tuned_model')

# -------------------------
# 3. FAISS INDEX SETUP
# -------------------------
def build_faiss_index(embeddings, output_index_path='movie_index.faiss'):
    """
    Build FAISS index for fast similarity search.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, output_index_path)

# -------------------------
# 4. MAIN FUNCTION
# -------------------------
def main():
    # Load and clean data
    df = pd.read_csv('data.csv')  # Replace with your file path
    
    # Optionally fine-tune the model
    model = fine_tune_model(df)
    
    # Generate embeddings
    embeddings = generate_embeddings(df)
    
    # Save embeddings and metadata
    np.save('movie_embeddings.npy', embeddings)
    df[['movie name', 'year', 'genres', 'poster_path']].to_csv('movie_metadata.csv', index=False)
    
    # Build FAISS index
    build_faiss_index(embeddings)

if __name__ == '__main__':
    main()
