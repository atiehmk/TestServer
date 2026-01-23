from flask import Flask, request

app = Flask(__name__)

@app.get("/")
def home():
    return "Hello! Simple Flask API is running.", 200

@app.get("/health")
def health():
    return {"status": "ok"}, 200

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data:
        return {"error": "No JSON received"}, 400
    return {"message": "Received", "received": data}, 200
