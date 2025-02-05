# app.py
# This Flask application serves the AI Market Predictor dashboard
# and provides API endpoints for generating predictions and retrieving historical predictions.

import os
from flask import Flask, render_template, jsonify
from predict import fetch_crypto_data, generate_market_prediction
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

app = Flask(__name__)

# Set up MongoDB connection.
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client['investment_db']
predictions_collection = db['predictions']

@app.route("/")
def dashboard():
    """Render the main dashboard."""
    return render_template("dashboard.html")

@app.route("/api/prediction")
def get_prediction():
    """API endpoint to generate a new market prediction."""
    crypto_data = fetch_crypto_data()
    prediction = generate_market_prediction(crypto_data)
    # Save the new prediction to MongoDB.
    predictions_collection.insert_one({"prediction": prediction})
    return jsonify({"prediction": prediction})

@app.route("/api/predictions")
def get_predictions():
    """API endpoint to retrieve the last 10 predictions."""
    preds = list(predictions_collection.find({}, {"_id": 0}).limit(10))
    return jsonify(preds)

if __name__ == '__main__':
    app.run(debug=True)