from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('heart_disease_model.joblib')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = int(request.form['age'])
        cp = int(request.form['cp'])
        thalach = int(request.form['thalach'])

        # Create a DataFrame for the input data
        user_data = pd.DataFrame({'age': [age], 'cp': [cp], 'thalach': [thalach]})
        
        # Make prediction
        prediction = model.predict(user_data)[0]
        result = "Heart Disease Present" if prediction == 1 else "No Heart Disease"
        
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
