import pickle
from sentence_transformers import CrossEncoder

model = CrossEncoder("finetune/cross_encoder_finetuned")
with open("embeddings/titles.pkl", "rb") as f:
    titles = pickle.load(f)

with open("embeddings/plots.pkl", "rb") as f:
    plots = pickle.load(f)

print("\n Welcome to the Fine-Tuned CrossEncoder Movie Agent")
print("Type a movie description or question. Type 'exit' to quit.\n")

while True:
    query = input("Your Query: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("Goodbye! ")
        break

    input_pairs = [[query, plot] for plot in plots]
    raw_scores = model.predict(input_pairs)
    scores = raw_scores[:, 1] if raw_scores.ndim == 2 else raw_scores

    ranked = sorted(zip(titles, plots, scores), key=lambda x: x[2], reverse=True)

    print("\n Top Matching Results:\n")
    for i, (title, plot, score) in enumerate(ranked[:5]):
        print(f"{i+1}. {title} (Score: {score:.4f})")
        print(f"   {plot[:300]}...\n")
