import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

# df = pd.read_csv("finetune/pairs.csv")
df = pd.read_csv("finetune/train.csv")
train_samples = []
for _, row in df.iterrows():
    train_samples.append(InputExample(texts=[row["query"], row["positive_plot"]]))

model = SentenceTransformer("all-MiniLM-L6-v2")
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=10,
    output_path="finetune/sbert_finetuned"
)

print("Model fine-tuned and saved to finetune/sbert_finetuned")
