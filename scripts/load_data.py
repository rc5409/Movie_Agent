import pandas as pd
import os
import pickle

def load_and_clean(csv_path="data/wiki_movie_plots_deduped.csv", max_entries=10000):
    df = pd.read_csv(csv_path)
    df = df[['Title', 'Plot']].dropna().drop_duplicates()
    df = df.head(max_entries)

    titles = df['Title'].tolist()
    plots = df['Plot'].tolist()

    os.makedirs("embeddings", exist_ok=True)

    with open("embeddings/titles.pkl", "wb") as f:
        pickle.dump(titles, f)

    with open("embeddings/plots.pkl", "wb") as f:
        pickle.dump(plots, f)

    print(f"Saved {len(titles)} titles and plots to embeddings/")

if __name__ == "__main__":
    load_and_clean()
