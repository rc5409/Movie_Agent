import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("finetune/pairs.csv")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
os.makedirs("finetune", exist_ok=True)
train_df.to_csv("finetune/train.csv", index=False)
test_df.to_csv("finetune/test.csv", index=False)

print(f" Saved {len(train_df)} training rows and {len(test_df)} test rows.")
