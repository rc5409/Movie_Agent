# scripts/embed_plots.py

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

def embed():
    with open("embeddings/plots.pkl", "rb") as f:
        plots = pickle.load(f)

    print(f"Encoding {len(plots)} plots using Sentence-BERT...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(plots, show_progress_bar=True, convert_to_numpy=True)
    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/movie_embeddings.npy", embeddings)

    print(f"Saved embeddings to embeddings/movie_embeddings.npy")

if __name__ == "__main__":
    embed()
