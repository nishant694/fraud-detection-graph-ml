import json, sqlite3, time, joblib, pandas as pd
from typing import Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel

B = "/content/drive/MyDrive/fraud detection project/"
model = joblib.load(B + "model_graph.pkl")
cfg = json.load(open(B + "config.json"))
COLS, THR = cfg["features"], cfg["threshold"]

con = sqlite3.connect("/content/fraud.db", check_same_thread=False)
con.execute("""CREATE TABLE IF NOT EXISTS predictions
  (ts REAL, transaction_id INTEGER, probability REAL, decision TEXT, latency_ms REAL)""")
con.commit()

app = FastAPI(title="Fraud Detection API")

class Tx(BaseModel):
    transaction_id: int
    features: Dict[str, Optional[float]]

@app.get("/health")
def health():
    return {"status": "ok", "threshold": THR}

@app.post("/predict")
def predict(tx: Tx):
    t0 = time.time()
    row = pd.DataFrame([tx.features]).reindex(columns=COLS).astype(float)
    p = float(model.predict_proba(row)[:, 1][0])
    decision = "FRAUD_ALERT" if p >= THR else "OK"
    ms = (time.time() - t0) * 1000
    con.execute("INSERT INTO predictions VALUES (?,?,?,?,?)",
                (time.time(), tx.transaction_id, p, decision, ms))
    con.commit()
    return {"transaction_id": tx.transaction_id,
            "fraud_probability": round(p, 4), "decision": decision}
