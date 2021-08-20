from flask import Flask, render_template, request, jsonify
import numpy as np
import sklearn.externals as extjoblib
import joblib

# Create Flask object to Run
app = Flask(__name__)

# Load the model from the File

premium_prediction = joblib.load('model/premium_pred_model.pkl')


@app.route('/', methods=['GET'])
def home():
    return "Insurance Premium Prediction!!!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    age = int(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']

    print(age)

    test_inp = np.array([age, sex, bmi, children, smoker, region]).reshape(1, 6)
    premium_predicted = int(premium_prediction.predict(test_inp)[0])
    output = "Predicted Insurance Premium: " + str(premium_predicted)

    return output


if __name__ == '__main__':
    app.run(debug=True)
