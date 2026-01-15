from flask import Flask, request, jsonify
import os

app = Flask(__name__)

SEQ_LEN = 30

def model_predict(events, metadata):
    # TODO: Prediction model code
    return {"state": "confused", "score": 0.5}

@app.get("/")
def home():
    return "API is running", 200

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Expected JSON"}), 400

    session_id = payload.get("id")
    metadata = payload.get("metadata", {})
    events = payload.get("events")

    if not session_id:
        return jsonify({"error": "Missing 'id'"}), 400
    if not isinstance(events, list):
        return jsonify({"error": "'events' must be a list"}), 400
    if len(events) != SEQ_LEN:
        return jsonify({"error": f"'events' must have exactly {SEQ_LEN} items", "got": len(events)}), 400

    prediction = model_predict(events, metadata)

    return jsonify({"received": True, "prediction": prediction}), 200

@app.get("/health")
def health():
    return "OK", 200

# Local-only runner (Render will ignore this because it uses gunicorn)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
