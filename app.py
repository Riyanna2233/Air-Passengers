# app.py
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the model and scaler
model = tf.keras.models.load_model('lstm_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(-1, 1)
    scaled_data = scaler.transform(input_data)
    
    time_step = 10
    if len(scaled_data) >= time_step:
        input_sequence = scaled_data[-time_step:]
        input_sequence = input_sequence.reshape(1, time_step, 1)
        prediction = model.predict(input_sequence)
        prediction = scaler.inverse_transform(prediction)
        return jsonify({'prediction': prediction.tolist()})
    else:
        return jsonify({'error': 'Input data is too short'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=4000)  # Change 8000 to the port number you want to use

