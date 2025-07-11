import pandas as pd
import os


TEST_PATH = "finetune/test.csv"
NEW_NAME  = "finetune/test_queries.csv"

def main() -> None:
    df = pd.read_csv(TEST_PATH)[["query", "positive_plot"]]
    os.makedirs(os.path.dirname(NEW_NAME), exist_ok=True)
    df.to_csv(NEW_NAME, index=False)
    print(f" Saved {len(df)} rows → {NEW_NAME}")

if __name__ == "__main__":
    main()
