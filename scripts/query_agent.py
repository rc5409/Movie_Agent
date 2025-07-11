# # scripts/query_agent.py

# import numpy as np
# import pickle
# from sentence_transformers import SentenceTransformer
# from scripts.build_index import build_index

# def query_movie_agent(query, top_k=3):
#     # Load movie titles and plots
#     with open("embeddings/titles.pkl", "rb") as f:
#         titles = pickle.load(f)
#     with open("embeddings/plots.pkl", "rb") as f:
#         plots = pickle.load(f)

#     # Load SBERT model
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     # Build index
#     index = build_index()

#     # Encode user query
#     query_embedding = model.encode([query]).astype('float32')

#     # Search index
#     distances, indices = index.search(query_embedding, top_k)

#     # Collect results
#     results = []
#     for idx in indices[0]:
#         results.append((titles[idx], plots[idx]))
    
#     return results


# scripts/query_agent.py

import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util
from scripts.build_index import build_index
import spacy

model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

def smart_rerank(results, query, top_k=5):
    query_doc = nlp(query)
    query_nouns = set([token.lemma_.lower() for token in query_doc if token.pos_ in ['NOUN', 'PROPN']])
    query_embedding = model.encode([query])[0]

    reranked = []
    for title, plot in results:
        plot_embedding = model.encode([plot])[0]
        semantic_score = util.pytorch_cos_sim(query_embedding, plot_embedding).item()
        plot_doc = nlp(plot)
        plot_nouns = set([token.lemma_.lower() for token in plot_doc if token.pos_ in ['NOUN', 'PROPN']])
        noun_overlap_score = len(query_nouns.intersection(plot_nouns)) / (len(query_nouns) + 1e-5)

        final_score = semantic_score + 0.2 * noun_overlap_score
        reranked.append((final_score, title, plot))

    reranked.sort(reverse=True)
    return [(title, plot) for score, title, plot in reranked[:top_k]]

def query_movie_agent(query, top_k=5):
    with open("embeddings/titles.pkl", "rb") as f:
        titles = pickle.load(f)
    with open("embeddings/plots.pkl", "rb") as f:
        plots = pickle.load(f)

    index = build_index()
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), top_k * 2)

    results = [(titles[i], plots[i]) for i in indices[0]]
    reranked_results = smart_rerank(results, query, top_k=top_k)

    return reranked_results

