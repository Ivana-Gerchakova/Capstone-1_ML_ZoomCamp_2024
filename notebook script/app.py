from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

rf_model = joblib.load('random_forest_model.pkl')  
nn_model = load_model('best_nn_model.keras') 
vectorizer = joblib.load('dict_vectorizer.pkl') 


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  
    input_features = np.array(data['features']).reshape(1, -1)  

    transformed_features = vectorizer.transform(input_features)

    rf_prediction = rf_model.predict(transformed_features).tolist()

    nn_prediction = nn_model.predict(transformed_features.toarray()).tolist()

    return jsonify({
        "random_forest_prediction": rf_prediction,
        "neural_network_prediction": nn_prediction
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

