import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm
import random
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
test_df = pd.read_csv("finetune/test_queries.csv")


with open("embeddings/plots.pkl", "rb") as f:
    all_plots = pickle.load(f)
plot2idx = {p: i for i, p in enumerate(all_plots)}

sbert_orig = SentenceTransformer("all-MiniLM-L6-v2", device=device)
sbert_tuned = SentenceTransformer("finetune/sbert_finetuned", device=device)
crossenc_tuned = CrossEncoder("finetune/cross_encoder_finetuned", device=device)

all_emb_orig = sbert_orig.encode(all_plots, convert_to_tensor=True, device=device, show_progress_bar=True)
all_emb_tuned = sbert_tuned.encode(all_plots, convert_to_tensor=True, device=device, show_progress_bar=True)

def build_pool(query, pos_plot, emb_model):
    pos_emb = emb_model.encode([pos_plot], convert_to_tensor=True, device=device)
    sims = util.cos_sim(pos_emb, all_emb_orig)[0].detach()
    top_idx = sims.topk(49).indices.tolist()
    top_idx = [i for i in top_idx if all_plots[i] != pos_plot]
    top_idx.append(plot2idx[pos_plot])
    random.shuffle(top_idx)
    return top_idx

def eval_sbert(model, all_emb):
    correct, mrr = 0, 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        q, pos = row["query"], row["positive_plot"]
        pool_idx = build_pool(q, pos, model)
        pool = [all_plots[i] for i in pool_idx]
        q_emb = model.encode([q], convert_to_tensor=True, device=device)
        pool_emb = all_emb[pool_idx]
        scores = util.cos_sim(q_emb, pool_emb)[0].detach().cpu().numpy()
        true_idx = pool_idx.index(plot2idx[pos])
        sorted_indices = scores.argsort()[::-1]
        rank = sorted_indices.tolist().index(true_idx)
        correct += rank == 0
        mrr += 1 / (rank + 1)
    return correct / len(test_df), mrr / len(test_df)

def eval_cross(model):
    correct, mrr = 0, 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        q, pos = row["query"], row["positive_plot"]
        pool_idx = build_pool(q, pos, sbert_orig)
        pool = [all_plots[i] for i in pool_idx]
        pairs = [[q, plot] for plot in pool]
        scores = model.predict(pairs)
        scores = scores[:, 1] if scores.ndim > 1 else scores  # softmax case
        true_idx = pool_idx.index(plot2idx[pos])
        rank = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True).index(true_idx)
        correct += rank == 0
        mrr += 1 / (rank + 1)
    return correct / len(test_df), mrr / len(test_df)


acc0, mrr0 = eval_sbert(sbert_orig, all_emb_orig)
acc1, mrr1 = eval_sbert(sbert_tuned, all_emb_tuned)
acc2, mrr2 = eval_cross(crossenc_tuned)

print("\nFinal Results:")
print(f"{'Model':<30} {'Accuracy@1':<10} {'MRR'}")
print(f"{'Original SBERT':<30} {acc0:.4f}      {mrr0:.4f}")
print(f"{'Fine-Tuned SBERT':<30} {acc1:.4f}      {mrr1:.4f}")
print(f"{'Fine-Tuned CrossEncoder':<30} {acc2:.4f}      {mrr2:.4f}")
