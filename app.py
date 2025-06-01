from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("house_price_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    size = float(request.form['size'])
    rooms = int(request.form['rooms'])
    type_input = request.form['type']

    type_encoded = 1 if type_input == 'individual' else 0
    features = np.array([[area, size, rooms, type_encoded]])
    prediction = model.predict(features)[0]
    return render_template('index.html', prediction_text=f'Estimated Price: ₹{prediction:.2f} lakhs')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

