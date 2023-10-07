from flask import Flask, request, render_template, jsonify
from pydantic import BaseModel
from flask_cors import CORS
import joblib

class PredictionInput(BaseModel):
    PetalLengthCm: float
    PetalWidthCm: float

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
model = joblib.load("randomForestIris.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    input_data = PredictionInput(**input_data)
    prediction = model.predict([[input_data.PetalLengthCm, input_data.PetalWidthCm]])[0]
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(port=8080)
