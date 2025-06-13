import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load your trained model once on startup
with open('house_price_prediction.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['CRIM']),
            float(request.form['ZN']),
            float(request.form['INDUS']),    # Fixed key name (uppercase)
            float(request.form['CHAS']),
            float(request.form['NOX']),
            float(request.form['RM']),
            float(request.form['AGE']),
            float(request.form['DIS']),       # Added DIS here
            float(request.form['RAD']),
            float(request.form['TAX']),
            float(request.form['PTRATIO']),
            float(request.form['B']),
            float(request.form['LSTAT']),
        ]

        features_array = np.array([features])
        prediction = model.predict(features_array)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f"Predicted Price: ${output}k")
    
    except Exception as e:
        # Return error message on invalid input or other errors
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
