import numpy as np
import faiss

def build_index(embedding_path="embeddings/movie_embeddings.npy"):
    embeddings = np.load(embedding_path).astype('float32')
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)  
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors of dimension {dim}")
    return index

if __name__ == "__main__":
    build_index()
