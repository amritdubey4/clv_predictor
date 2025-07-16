from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('clv_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X = pd.DataFrame([[
        data['avg_purchase'],
        data['frequency'],
        data['churn_rate']
    ]], columns=['avg_purchase_value', 'purchase_frequency', 'churn_rate'])
    clv = model.predict(X)[0]
    return jsonify({'clv': clv})

if __name__ == '__main__':
    app.run(debug=True)