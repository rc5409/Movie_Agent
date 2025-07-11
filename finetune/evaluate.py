import random, pickle, time, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def load_corpus():
    titles = pickle.load(open("embeddings/titles.pkl", "rb"))
    plots  = pickle.load(open("embeddings/plots.pkl",  "rb"))
    return titles, plots

def accuracy_mrr(model, df, titles, plots, pool_size=1000):
    acc, rr = 0, []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        q, pos_plot = row["query"], row["positive_plot"]

        
        pool_idx = random.sample(range(len(plots)), pool_size-1)
        pool_texts = [plots[i] for i in pool_idx] + [pos_plot]

        emb = model.encode([q] + pool_texts, convert_to_tensor=True)
        sims = util.cos_sim(emb[0], emb[1:])[0].cpu().numpy()
        rank = len(sims) - sims.argsort().argsort()[-1]  
        acc += rank == 1
        rr.append(1.0 / rank)
    return acc/len(df), np.mean(rr)

df_test = pd.read_csv("finetune/test.csv")
titles, plots = load_corpus()

for name, path in [
    ("Original SBERT", "all-MiniLM-L6-v2"),
    ("Fine-tuned SBERT", "finetune/sbert_finetuned"),
]:
    mdl = SentenceTransformer(path)
    t0 = time.time()
    acc, mrr = accuracy_mrr(mdl, df_test, titles, plots)
    print(f"{name:<18} | Acc@1: {acc:.3f} | MRR: {mrr:.3f} | Δt: {time.time()-t0:.1f}s")
