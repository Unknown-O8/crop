from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model and label encoder
model = pickle.load(open("crop_recommendation.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract input values
        features = [
            data['nitrogen'], data['phosphorus'], data['potassium'],
            data['temperature'], data['humidity'], data['ph'],
            data['rainfall']
        ]

        # Convert to NumPy array and reshape for model
        input_data = np.array(features).reshape(1, -1)

        # Get prediction
        prediction = model.predict(input_data)[0]

        # Convert numerical prediction back to crop name
        crop_name = label_encoder.inverse_transform([prediction])[0]

        # Send response
        return jsonify({'crop': crop_name})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
