import sqlite3
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Monitor", layout="wide")
st.title("Fraud Monitoring Dashboard")

# Load logged predictions and remove duplicates
log = pd.read_sql("SELECT * FROM predictions", sqlite3.connect("fraud.db"))
log = log.drop_duplicates("transaction_id", keep="last")
labels = pd.read_parquet("replay_sample.parquet", columns=["TransactionID", "isFraud"])
df = log.merge(labels, left_on="transaction_id", right_on="TransactionID", how="left")

# Adjustable alert threshold
st.sidebar.header("Controls")
thr = st.sidebar.slider("Alert threshold", 0.0, 1.0, 0.82, 0.01)
df["alert"] = df["probability"] >= thr

# KPIs
tp = int((df.alert & (df.isFraud == 1)).sum())
fp = int((df.alert & (df.isFraud == 0)).sum())
fn = int((~df.alert & (df.isFraud == 1)).sum())
precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transactions scored", len(df))
c2.metric("Alerts raised", int(df.alert.sum()))
c3.metric("Precision", f"{precision:.2f}")
c4.metric("Recall", f"{recall:.2f}")
c5.metric("Latency p95 (ms)", f"{df.latency_ms.quantile(.95):.1f}")

# Charts
left, right = st.columns(2)
with left:
    st.subheader("Fraud probability distribution")
    hist = pd.cut(df.probability, bins=20).value_counts().sort_index()
    hist.index = hist.index.astype(str)
    st.bar_chart(hist)
with right:
    st.subheader("Alerts over time (batches of 50)")
    df["batch"] = df.index // 50
    st.line_chart(df.groupby("batch")["alert"].sum())

# Alert table
st.subheader("Top suspicious transactions")
top = df.sort_values("probability", ascending=False).head(20)
st.dataframe(top[["transaction_id", "probability", "decision", "isFraud", "latency_ms"]].round(3),
             use_container_width=True)
