# Movie Plot Retrieval Agent

This project implements a semantic search system that retrieves the most relevant movie plot given a natural language query. It compares two retrieval paradigms:

- **Bi-Encoder (Sentence-BERT)**
  Uses pretrained SBERT (all-MiniLM-L6-v2).

    Builds a FAISS index over the corpus (plots.pkl) for fast semantic search.

    Retrieves top-K similar results using cosine similarity in embedding space.

    Re-ranks results using:

    SBERT cosine similarity again (on query and individual plot).

    Noun-overlap score using spaCy.

    Final score = semantic score + 0.2 * noun overlap.
- **Fine-Tuned SBERT Query with Re-ranking**
    Uses fine-tuned SBERT trained with MultipleNegativesRankingLoss.

    Encodes movie plots and builds FAISS index once with tuned embeddings.

    Queries are encoded with fine-tuned SBERT, so results better match semantic intent.

    Re-ranks retrieved results using the same smart_rerank logic.
  
- **CrossEncoder (fine-tuned for classification)**
    The CrossEncoder does not precompute embeddings for individual plots.

    Instead, it takes (query, plot) pairs and computes a score for each pair jointly.

    This joint scoring enables fine-grained interaction between the query and the text (e.g., attention across tokens), making it more expressive.

    The goal is to evaluate trade-offs between latency and retrieval accuracy using real-world movie plot data.

## Project Overview

Given a user-provided query like  
_"a robot who learns to love"_  
the system retrieves the most semantically relevant movie plot from a large dataset of Wikipedia movie summaries.

This project tests the system’s ability to understand the semantics of both the query and the movie plots, making it ideal for evaluating LLM-style sentence embeddings and inference strategies.

## Project Structure

```
movie_ai_agent/
├── data/
│   └── wiki_movie_plots_deduped.csv
├── embeddings/
│   ├── plots.pkl
│   └── titles.pkl
├── finetune/
│   ├── sbert_finetuned/
│   ├── cross_encoder_finetuned/
│   ├── contrastive_sbert.py
│   ├── cross_encoder.py
│   ├── evaluate_all.py
│   ├── evaluate.py
│   ├── pairs.csv
│   ├── test_queries.csv
│   ├── test.csv
│   └── train.csv
├── scripts/
│   ├── build_index.py
│   ├── embed_plots.py
│   ├── evaluation.py
│   ├── generate_pairs.py
│   ├── load_data.py
│   ├── query_agent.py
│   ├── regenerate_pairs.py
│   ├── run_agent.py
│   ├── run_finetune_cross_encoder.py
│   └── split_train_test.py
├── combined_metrics.png
├── latency_comparison.png
├── latency.png
├── performance_metrics.png
├── roc_curve.png
├── requirements.txt
└── README.md
```

### Dataset

This project uses the [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) dataset, which contains over 34,000 movie titles along with their plot summaries, genres, and metadata. For this project, only the **movie titles** and **plot summaries** were used.

#### Preprocessing

- Extracted and cleaned movie `titles` and `plots`, stored as `titles.pkl` and `plots.pkl` under the `embeddings/` folder.
- All text was lowercased and stripped of special characters to maintain consistency.
- These `.pkl` files are used across all embedding and inference scripts.

#### Test Query Generation

The test set is defined in `finetune/test_queries.csv`, which contains `query` and `positive_plot` columns. It was generated using the `scripts/regenerate_pairs.py` script:

```bash
python scripts/regenerate_pairs.py --mode test
```

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Embed Movie Plots

```bash
python scripts/embed_plots.py
```

### 3. Generate Training Pairs

```bash
python scripts/generate_pairs.py
```

### 4. Train the Models

```bash
python finetune/contrastive_sbert.py
python finetune/cross_encoder.py
```

### 5. Generate Positive-Only Test Pairs

```bash
python scripts/regenerate_pairs.py
```

### 6. Evaluate

```bash
python scripts/evaluation.py
```

### 7. Query Agents

```bash
python scripts/run_agent #generic model
python scripts/run_finetune_sbert.py                  # SBERT-based
python scripts/run_finetune_cross_encoder.py # CrossEncoder-based
```

## Evaluation Metrics

| Model                    | Accuracy@1 ± SD     | MRR ± SD          | Latency (ms) | AUC     |
|-------------------------|---------------------|-------------------|--------------|---------|
| Original SBERT          | 0.9187 ± 0.2732     | 0.9453 ± 0.1900   | 0.23         | 0.9746  |
| Fine-Tuned SBERT        | 0.9938 ± 0.0788     | 0.9946 ± 0.0676   | 0.23         | 0.9999  |
| Fine-Tuned CrossEncoder | 0.9938 ± 0.0788     | 0.9969 ± 0.0394   | 265.71       | 1.0000  |

## Visualizations

- `performance_metrics.png`: Accuracy@1 and MRR comparison
- `latency_comparison.png`: Inference time across models
- `roc_curve.png`: ROC-AUC comparison
- `combined_metrics.png`: Metrics with standard deviation bars

## Notes on Test Design

- `regenerate_pairs.py` creates test queries with **no hard negatives**.
- During evaluation, **hard negatives are dynamically sampled** using cosine similarity.
- This helps isolate model quality without overlap or contamination.

## Conclusions

- **SBERT (fine-tuned)** is suitable for real-time use due to high accuracy and low latency.
- **CrossEncoder** provides highest precision but much slower — suitable for reranking.
- AUC and MRR improvements clearly show the benefit of fine-tuning both approaches.
