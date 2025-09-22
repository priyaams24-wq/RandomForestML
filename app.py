from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    data = request.get_json()

    # Convert the input features into a NumPy array
    features = np.array(data['features']).reshape(1, -1)

    # Predict the class using the trained model
    prediction = model.predict(features)

    species = {0: "setosa", 1: "versicolor", 2: "virginica"}
    predicted_label = species[int(prediction[0])]
    # Return the prediction as JSON
    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
