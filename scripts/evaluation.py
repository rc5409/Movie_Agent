import torch, time, pickle, random
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

device = "cuda" if torch.cuda.is_available() else "cpu"
test_df = pd.read_csv("finetune/test_queries.csv")

with open("embeddings/plots.pkl", "rb") as f:
    all_plots = pickle.load(f)
with open("embeddings/titles.pkl", "rb") as f:
    all_titles = pickle.load(f)

plot2idx = {p: i for i, p in enumerate(all_plots)}

sbert_orig = SentenceTransformer("all-MiniLM-L6-v2", device=device)
sbert_tuned = SentenceTransformer("finetune/sbert_finetuned", device=device)
crossenc_tuned = CrossEncoder("finetune/cross_encoder_finetuned", device=device)

emb_orig = sbert_orig.encode(all_plots, convert_to_tensor=True, device=device)
emb_tuned = sbert_tuned.encode(all_plots, convert_to_tensor=True, device=device)

results = {
    "Model": [],
    "Accuracy@1": [],
    "Accuracy_std": [],
    "MRR": [],
    "MRR_std": [],
    "Latency (ms)": [],
    "AUC": []
}

roc_curves = {}

def build_pool(pos_plot, all_emb):
    pos_emb = all_emb[plot2idx[pos_plot]].unsqueeze(0)
    sims = util.cos_sim(pos_emb, all_emb)[0].cpu().numpy()
    idxs = np.argsort(sims)[::-1]
    hard_negs = [i for i in idxs if all_plots[i] != pos_plot][:49]
    hard_negs.append(plot2idx[pos_plot])
    random.shuffle(hard_negs)
    return hard_negs

def eval_sbert(model, name, all_emb):
    acc, mrr, latency = [], [], []
    y_true, y_scores = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        q, pos = row["query"], row["positive_plot"]
        pool_idx = build_pool(pos, all_emb)
        pool = [all_plots[i] for i in pool_idx]
        true_idx = pool_idx.index(plot2idx[pos])
        q_emb = model.encode(q, convert_to_tensor=True, device=device)
        pool_emb = all_emb[pool_idx]
        start = time.time()
        scores = util.cos_sim(q_emb, pool_emb)[0].cpu().numpy()
        latency.append((time.time() - start) * 1000)
        sorted_idx = np.argsort(scores)[::-1]
        rank = np.where(sorted_idx == true_idx)[0][0]
        acc.append(rank == 0)
        mrr.append(1 / (rank + 1))
        y = [0] * len(pool)
        y[true_idx] = 1
        y_true += y
        y_scores += list(scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_curves[name] = (fpr, tpr)
    results["Model"].append(name)
    results["Accuracy@1"].append(np.mean(acc))
    results["Accuracy_std"].append(np.std(acc))
    results["MRR"].append(np.mean(mrr))
    results["MRR_std"].append(np.std(mrr))
    results["Latency (ms)"].append(np.mean(latency))
    results["AUC"].append(roc_auc_score(y_true, y_scores))

def eval_cross(name):
    acc, mrr, latency = [], [], []
    y_true, y_scores = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        q, pos = row["query"], row["positive_plot"]
        pool_idx = build_pool(pos, emb_orig)
        pool = [all_plots[i] for i in pool_idx]
        true_idx = pool_idx.index(plot2idx[pos])
        pairs = [[q, p] for p in pool]
        start = time.time()
        scores = crossenc_tuned.predict(pairs)
        scores = scores[:, 1] if scores.ndim == 2 else scores
        latency.append((time.time() - start) * 1000)
        sorted_idx = np.argsort(scores)[::-1]
        rank = np.where(sorted_idx == true_idx)[0][0]
        acc.append(rank == 0)
        mrr.append(1 / (rank + 1))
        y = [0] * len(pool)
        y[true_idx] = 1
        y_true += y
        y_scores += list(scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_curves[name] = (fpr, tpr)
    results["Model"].append(name)
    results["Accuracy@1"].append(np.mean(acc))
    results["Accuracy_std"].append(np.std(acc))
    results["MRR"].append(np.mean(mrr))
    results["MRR_std"].append(np.std(mrr))
    results["Latency (ms)"].append(np.mean(latency))
    results["AUC"].append(roc_auc_score(y_true, y_scores))

eval_sbert(sbert_orig, "Original SBERT", emb_orig)
eval_sbert(sbert_tuned, "Fine-Tuned SBERT", emb_tuned)
eval_cross("Fine-Tuned CrossEncoder")

print("\n=== Final Results ===")
for i in range(len(results["Model"])):
    print(f"{results['Model'][i]:<25} | "
          f"Accuracy@1: {results['Accuracy@1'][i]:.4f} ± {results['Accuracy_std'][i]:.4f} | "
          f"MRR: {results['MRR'][i]:.4f} ± {results['MRR_std'][i]:.4f} | "
          f"Latency: {results['Latency (ms)'][i]:.2f} ms | "
          f"AUC: {results['AUC'][i]:.4f}")

x = np.arange(len(results["Model"]))
bar_width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - bar_width, results["Accuracy@1"], width=bar_width, label="Accuracy@1", yerr=results["Accuracy_std"])
plt.bar(x, results["MRR"], width=bar_width, label="MRR", yerr=results["MRR_std"])
plt.bar(x + bar_width, results["AUC"], width=bar_width, label="AUC")
plt.xticks(x, results["Model"])
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("combined_metrics.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(results["Model"], results["Latency (ms)"], color="orange")
plt.ylabel("Latency (ms)")
plt.title("Average Inference Time")
plt.tight_layout()
plt.savefig("latency.png")
plt.show()

plt.figure(figsize=(8, 6))
for model, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{model} (AUC={results['AUC'][results['Model'].index(model)]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
