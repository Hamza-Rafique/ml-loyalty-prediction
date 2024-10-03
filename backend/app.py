# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS 
import pickle
import numpy as np

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app) 
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array([[
        data['Frequency_of_Communication'],
        data['Help_in_Crises'],
        data['Financial_Support_Provided'],
        data['Attendance_at_Events'],
        data['Sentiment_Score']
    ]])
    
    prediction = model.predict(input_data)[0]
    
    return jsonify({'loyalty_score': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
