from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from xgboost import XGBRegressor  # Ensure this is present

app = Flask(__name__)
CORS(app)

# Load the model
model = joblib.load('calorie_burnt_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        data['gender'], 
        data['age'], 
        data['height'], 
        data['weight'], 
        data['duration'], 
        data['heart_rate'], 
        data['body_temp']
    ]
    prediction = model.predict([features])
    return jsonify({'prediction': float(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
