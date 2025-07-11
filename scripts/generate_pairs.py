import os
import pickle
import random
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
with open(os.path.join(BASE_DIR, "embeddings/titles.pkl"), "rb") as f:
    titles = pickle.load(f)
with open(os.path.join(BASE_DIR, "embeddings/plots.pkl"), "rb") as f:
    plots = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
MAX = 300
titles = titles[:MAX]
plots = plots[:MAX]
def generate_query_from_plot(plot):
    templates = [
        "What's the movie where {}",
        "Find the movie in which {}",
        "Which film features a story where {}",
        "Give me a plot about how {}",
        "Movie where {}"
    ]
    sentences = plot.split(".")
    if not sentences or sentences[0].strip() == "":
        return plot[:200]
    base = sentences[0].strip().lower()
    return random.choice(templates).format(base)


rows = []
for i in tqdm(range(len(plots))):
    query = generate_query_from_plot(plots[i])
    pos = plots[i]

    while True:
        j = random.randint(0, len(plots) - 1)
        if j != i:
            neg = plots[j]
            sim = util.cos_sim(model.encode([query]), model.encode([neg]))[0][0].item()
            if sim < 0.3:
                break

    rows.append({
        "query": query,
        "positive_plot": pos,
        "negative_plot": neg
    })

os.makedirs(os.path.join(BASE_DIR, "finetune"), exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(BASE_DIR, "finetune/pairs.csv"), index=False)
print(f"Saved {len(df)} template-based query pairs to finetune/pairs.csv")
