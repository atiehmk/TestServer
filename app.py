from flask import Flask, request, jsonify
import os
import numpy as np
import pandas as pd
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import tensorflow as tf
import joblib
from collections import defaultdict, deque
import json

app = Flask(__name__)

# ===== config =====
SEQ_LEN = 30

# Model outputs these 6 continuous labels
emotion_labels = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Confusion']

# ---- load model + artifacts once ----
MODEL_PATH = os.environ.get("MODEL_PATH", "best_lstm_sliding_split.keras")
FEATURE_COLUMNS_PATH = os.environ.get("FEATURE_COLUMNS_PATH", "feature_columns.pkl")
LABEL_SCALER_PATH = os.environ.get("LABEL_SCALER_PATH", "lab_scaler.pkl")

model = tf.keras.models.load_model(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
lab_scaler = joblib.load(LABEL_SCALER_PATH) if os.path.exists(LABEL_SCALER_PATH) else None

# Columns to one-hot encode (must match what you did when building aligned_df)
cols_to_encode = [
    'type', 'content.description', 'content.header', 'content.numUserPlacedComponents',
    'content.simType', 'content.status', 'content.subtitle', 'content.title',
    'element.locked', 'element.passed', 'element.spec', 'element.type',
    'element.userPlaced', 'idle', 'level', 'message', 'step.content.title',
    'step.position.relative', 'step.requiredAction.target', 'step.requiredAction.type'
]

# ---- in-memory buffers (per session/user) ----
sessions = defaultdict(lambda: deque(maxlen=SEQ_LEN))


def deep_get(d, path, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def flatten_event(ev: dict, metadata: dict | None = None) -> dict:
    """
    Convert one raw event into the flat feature fields used for one-hot encoding.
    Keep this aligned with your training preprocessing.
    """
    flat = {}

    # direct fields
    flat["type"] = ev.get("type")
    flat["message"] = ev.get("message")

    # In your raw, "level" is often in ev["name"] for BEGIN/FINISH events
    flat["level"] = ev.get("name")

    # idle (if you had a real idle computation in training, replicate it here)
    flat["idle"] = ev.get("idle", False)

    # content.*
    flat["content.title"] = deep_get(ev, "content.title")
    flat["content.subtitle"] = deep_get(ev, "content.subtitle")
    flat["content.header"] = deep_get(ev, "content.header")
    flat["content.description"] = deep_get(ev, "content.description")
    flat["content.simType"] = deep_get(ev, "content.simType")
    flat["content.status"] = deep_get(ev, "content.status")
    flat["content.numUserPlacedComponents"] = deep_get(ev, "content.numUserPlacedComponents")

    # step.*
    flat["step.content.title"] = deep_get(ev, "step.content.title")
    flat["step.position.relative"] = deep_get(ev, "step.position.relative")
    flat["step.requiredAction.target"] = deep_get(ev, "step.requiredAction.target")
    flat["step.requiredAction.type"] = deep_get(ev, "step.requiredAction.type")

    # element.* (may not exist on many events)
    flat["element.locked"] = deep_get(ev, "element.locked")
    flat["element.passed"] = deep_get(ev, "element.passed")
    flat["element.spec"] = deep_get(ev, "element.spec")
    flat["element.type"] = deep_get(ev, "element.type")
    flat["element.userPlaced"] = deep_get(ev, "element.userPlaced")

    return flat


def make_features_from_events(events: list[dict], metadata: dict | None = None) -> np.ndarray:
    """
    events: list of length SEQ_LEN
    returns: X of shape (1, SEQ_LEN, n_features) aligned to feature_columns.pkl
    """
    rows = [flatten_event(e, metadata) for e in events]
    df = pd.DataFrame(rows)

    # one-hot only the intended categorical columns that are present
    cols_present = [c for c in cols_to_encode if c in df.columns]
    df = pd.get_dummies(df, columns=cols_present, dummy_na=False, dtype="int8")

    # hard-align to training columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    X = df.to_numpy(dtype=np.float32)

    if X.shape[0] != SEQ_LEN:
        raise ValueError(f"Need exactly {SEQ_LEN} events after processing, got {X.shape[0]}")

    return np.expand_dims(X, axis=0)


def decode_prediction(y_pred):
    """
    LSTM outputs 6 continuous values in [0,1] (scaled space).
    If lab_scaler.pkl exists, also return inverse-scaled 'original' values.
    """
    y = np.array(y_pred)

    # If (1, T, 6) -> take last timestep
    if y.ndim == 3:
        y = y[:, -1, :]
    y = np.squeeze(y)  # -> (6,)

    if y.ndim != 1:
        return {"raw": y.astype(float).tolist()}

    scaled = y.astype(float)

    result = {
        "labels": emotion_labels,
        "scaled": scaled.tolist(),
        "scaled_by_label": {emotion_labels[i]: float(scaled[i]) for i in range(len(emotion_labels))}
    }

    if lab_scaler is not None and len(scaled) == len(emotion_labels):
        orig = lab_scaler.inverse_transform(scaled.reshape(1, -1)).reshape(-1)
        result["original"] = orig.astype(float).tolist()
        result["original_by_label"] = {emotion_labels[i]: float(orig[i]) for i in range(len(emotion_labels))}

    return result


def predict_from_events(events: list[dict], metadata: dict | None = None):
    X = make_features_from_events(events, metadata)
    y_pred = model.predict(X, verbose=0)
    return decode_prediction(y_pred)


@app.get("/")
def home():
    return "API is running", 200


@app.get("/health")
def health():
    return "OK", 200


# ---- Option 1: send a batch of 30+ events ----
@app.post("/predict")
def predict_batch():
    data = request.get_json(silent=True)
    #print("PREDICT data:", json.dumps(data, ensure_ascii=False)[:500])
    print("PREDICT id:", data.get("id"), "events:", len(data.get("events", [])))

    if not data:
        return jsonify({"error": "Expected JSON"}), 400

    session_id = data.get("id")
    events = data.get("events")
    metadata = data.get("metadata", {})

    if not session_id:
        return jsonify({"error": "Missing 'id'"}), 400
    if not isinstance(events, list):
        return jsonify({"error": "'events' must be a list"}), 400
    if len(events) < SEQ_LEN:
        return jsonify({"error": f"Need at least {SEQ_LEN} events", "got": len(events)}), 400

    # IMPORTANT: ensure correct time/sequence order before taking last 30
    events = sorted(events, key=lambda e: e.get("order", e.get("created", 0)))
    events = events[-SEQ_LEN:]

    try:
        pred = predict_from_events(events, metadata)
    except Exception as e:
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500

    player_id = (metadata or {}).get("playerId")
    return jsonify({"id": session_id, "playerId": player_id, "prediction": pred}), 200


# ---- Option 2: real-time ingest 1 event at a time, predict whenever buffer hits 30 ----
@app.post("/ingest")
def ingest_event():
    data = request.get_json(silent=True)
    #print("INGEST data:", json.dumps(data, ensure_ascii=False)[:500])
    print("INGEST id:", data.get("id"), "events:", len(data.get("events", [])))

    if not data:
        return jsonify({"error": "Expected JSON"}), 400

    session_id = data.get("id")
    event = data.get("event")  # single event dict
    metadata = data.get("metadata", {})

    if not session_id:
        return jsonify({"error": "Missing 'id'"}), 400
    if not isinstance(event, dict):
        return jsonify({"error": "'event' must be an object"}), 400

    buf = sessions[session_id]
    buf.append(event)

    player_id = (metadata or {}).get("playerId")

    # not enough yet
    if len(buf) < SEQ_LEN:
        return jsonify({"id": session_id, "playerId": player_id, "buffered": len(buf), "ready": False}), 200

    # ready -> predict on last 30 in arrival order (assumes events arrive in order)
    try:
        pred = predict_from_events(list(buf), metadata)
    except Exception as e:
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500

    return jsonify({"id": session_id, "playerId": player_id, "buffered": len(buf), "ready": True, "prediction": pred}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
