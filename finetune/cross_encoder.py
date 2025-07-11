import pandas as pd
from sentence_transformers import InputExample, CrossEncoder
from torch.utils.data import DataLoader


df = pd.read_csv("finetune/train.csv")
train_samples = []
for _, row in df.iterrows():
    train_samples.append(InputExample(texts=[row["query"], row["positive_plot"]], label=1.0))
    train_samples.append(InputExample(texts=[row["query"], row["negative_plot"]], label=0.0))

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    num_labels=2,
    automodel_args={"ignore_mismatched_sizes": True}
)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
model.fit(
    train_dataloader=train_dataloader,
    epochs=2,
    warmup_steps=100,
    output_path="finetune/cross_encoder_finetuned"
)

model.save("finetune/cross_encoder_finetuned")
