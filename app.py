from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import logging

# Application logging
# Configuring logging operations

logging.basicConfig(filename='app_deployment_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')

# Create Flask object to Run
app = Flask(__name__)

# Load the model from the File

model_load = joblib.load('./model/premium_pred_model.pkl')

logging.info('Pickle file loading completed.')


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        bmi = request.form['bmi']
        region_northwest = request.form['region_northwest']
        region_southeast = request.form['region_southeast']
        region_southwest = request.form['region_southwest']
        sex = request.form['sex']
        smoker = request.form['smoker']
        children = request.form['children']
        input_val = [age,
                     bmi,
                     region_northwest,
                     region_southeast,
                     region_southwest,
                     sex,
                     smoker,
                     children]
        final_features = [np.array(input_val)]

        output = model_load.predict(final_features).tolist()

        # logging operation
        logging.info(f"Insurance Premium is {output}")

        logging.info('Prediction getting posted to the web page.')

        return render_template('index.html', prediction_text=f'Insurance Premium is $ {output} ')
    else:
        return render_template('index.html')


@app.route("/predict_api", methods=['POST', 'GET'])
def predict_api():
    print(" request.method :", request.method)
    if request.method == 'POST':
        age = request.json['age']
        bmi = request.json['bmi']
        region_northwest = request.json['region_northwest']
        region_southeast = request.json['region_southeast']
        region_southwest = request.json['region_southwest']
        sex = request.json['sex']
        smoker = request.json['smoker']
        children = request.json['children']
        input_val = [age,
                     bmi,
                     region_northwest,
                     region_southeast,
                     region_southwest,
                     sex,
                     smoker,
                     children]
        final_features = [np.array(input_val)]

        output = model_load.predict(final_features).tolist()
        return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
