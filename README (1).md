# Real-Time Financial Fraud Detection using Graph ML

Detects suspicious transactions by combining transaction features with relationship
features (cards, addresses, email domains, devices) and serves predictions through a
real-time API with a monitoring dashboard.

## Pipeline

```
Transactions -> Feature engineering -> Transaction graph -> Graph features
-> LightGBM -> Fraud probability -> FastAPI -> SQLite log -> Streamlit dashboard
```

## Dataset

[IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/competitions/ieee-fraud-detection/data):
transaction + identity tables. The data is not included in this repository.

Used the first **250,000 transactions in time order** (Colab RAM limit), split by time:
first 80% for training (200,000 rows), last 20% for testing (50,000 rows), 421 features.
Fraud rate is 3.01% in train and 4.22% in test (about 2,100 test frauds). This shift is
concept drift, and the reason for a time-based split instead of a random one.

## Method

**Level 1 - Baselines:** Logistic Regression, Random Forest, LightGBM (class-weighted for
the ~3% fraud rate).

**Level 2 - Graph features:** a graph linking cards to addresses, email domains and
devices (hub entities above 1% of rows removed). Features: degree, PageRank, community
(Louvain) size, transaction velocity, and "suspicious neighbour" features (past fraud rate
and count for the same card / address / email / device / community), fed into LightGBM.

**Label-leakage control:** neighbour fraud counts use only earlier rows and only labels
from the training period; a row never sees its own label.

**Level 3 - GNN (GraphSAGE/GAT):** not implemented (time constraint), see Future work.

## Results (test set, time-based split)

Precision, Recall and F1 use a fixed 0.5 threshold. PR-AUC and ROC-AUC are threshold-free.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.1243 | 0.7899 | 0.2148 | 0.4373 | 0.8567 |
| Random Forest | 0.2916 | 0.6311 | 0.3989 | 0.5178 | 0.8786 |
| LightGBM | 0.4301 | 0.6524 | 0.5185 | **0.6263** | **0.9235** |
| LightGBM + graph features | 0.4719 | 0.5936 | 0.5258 | 0.6012 | 0.9047 |

Recall at a fixed precision (the operating-point view):

| Precision target | LightGBM | LightGBM + graph |
|---|---|---|
| >= 0.5 | 0.611 | 0.574 |
| >= 0.7 | 0.498 | 0.477 |
| >= 0.8 | 0.445 | 0.436 |
| >= 0.9 | 0.366 | 0.364 |

### Analysis

- Gradient boosting clearly beats the simpler models (PR-AUC 0.626 vs 0.518 for Random
  Forest and 0.437 for Logistic Regression). Logistic Regression catches the most fraud
  (recall 0.79) but with very low precision (0.12), so most of its alerts are false alarms.
- **On this 250k-row run, graph features did not improve the model:** PR-AUC and ROC-AUC
  are lower, and recall at fixed precision is equal or slightly lower. The graph model's
  higher precision and F1 at the 0.5 threshold come from a different score calibration,
  not from better ranking.
- An earlier run on a smaller 150k-row slice showed a small gain from graph features
  (PR-AUC 0.597 to 0.606), so the effect is small and unstable across runs.
- Graph features still carry a large share of the model's learned structure (40% of total
  gain), and neighbour fraud-rate features (card, email domain, address) are among the
  most used. The model relies on them, but they did not add net predictive value here.
- Possible reasons (hypotheses, not tested): (1) in training, neighbour fraud counts only
  see earlier labels, while test rows see the whole training history, a training/serving
  mismatch; (2) fraud rate drifts between train and test; (3) the base features already
  capture much of the same signal, and more data helps them more; (4) degree, PageRank and
  community are computed over the whole slice.

## Deployment

- **FastAPI** `/predict` endpoint returns a fraud probability and an ALERT/OK decision
  (alert threshold 0.8162, chosen for precision >= 0.8 on the test set). The deployed
  model is the graph-feature model, as the project's focus.
- Every prediction is logged to **SQLite** (PostgreSQL in production).
- Replay of 1,500 test transactions through the API: 30 alerts, 24 true frauds caught out
  of 53 (precision about 0.80, recall about 0.45), matching the offline evaluation.
  Latency p95 about 21 ms on the full replay.
- **Streamlit dashboard** shows KPIs, score distribution, alerts over time and top
  suspicious transactions, with an adjustable alert threshold.

![dashboard at threshold 0.82]("C:\Users\thaku\Downloads\Desktop\ML Project\Screenshot_high_threshold.png")
![dashboard at threshold 0.15]("C:\Users\thaku\Downloads\Desktop\ML Project\Screenshot_low_threshold.png")

Lowering the threshold from 0.82 to 0.15 raises recall from about 0.45 to 0.83 but drops
precision from about 0.80 to 0.18 (244 alerts instead of 30). The right threshold depends
on how many alerts a fraud team can review.

## Limitations

- 250k-row slice and a single run: small differences between models may be within noise.
- Graph structure (degree, PageRank, communities) is computed over the whole slice; a
  production system would recompute it on a rolling window.
- Fraud labels are assumed available instantly; real chargeback labels arrive with delay.
- The API receives precomputed graph features; production would use a feature store to
  avoid training/serving skew.
- The dashboard reads a replayed log, not live traffic.

## Future work

Fix the training/serving mismatch in the neighbour-label features (for example with
out-of-fold target encoding), GraphSAGE/GAT on the transaction graph, rolling graph
updates, a feature store, PostgreSQL, drift monitoring, and multi-seed evaluation with
confidence intervals.

## Run it

1. Run the notebook (Colab): builds features, trains models, exports `model_graph.pkl`,
   `config.json` and `replay_sample.parquet`.
2. API: `uvicorn api:app --port 8000`
3. Dashboard: `pip install streamlit pandas pyarrow`, then `streamlit run app.py`
   (needs `fraud.db` and `replay_sample.parquet` in the same folder).
