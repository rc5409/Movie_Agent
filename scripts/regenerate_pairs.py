import os, random, pickle, pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


titles = pickle.load(open(f"{BASE}/embeddings/titles.pkl", "rb"))
plots  = pickle.load(open(f"{BASE}/embeddings/plots.pkl",  "rb"))
corpus = list(zip(titles, plots))

sbert = SentenceTransformer("all-MiniLM-L6-v2")

TEMPLATES = [
    "Which movie tells the story where {}?",
    "Find the film about how {}.",
    "What’s the movie in which {}?",
    "Name the movie where {}"
]

def make_query(plot):
    sent = plot.split(".")[0].strip().lower()
    return random.choice(TEMPLATES).format(sent)

rows, MAX = [], 800                       
for i in tqdm(range(MAX)):
    q, pos = make_query(plots[i]), plots[i]
    sims = util.cos_sim(
        sbert.encode([pos]), sbert.encode(plots[:2000])
    )[0].cpu().numpy()
    hard_idx = sims.argsort()[-10:-1]   
    neg = plots[random.choice(hard_idx)]

    rows.append({"query": q, "positive_plot": pos, "negative_plot": neg})

df = pd.DataFrame(rows)
os.makedirs(f"{BASE}/finetune", exist_ok=True)
df.to_csv(f"{BASE}/finetune/pairs.csv", index=False)
print(f" Generated {len(df)} challenging triples → finetune/pairs.csv")
