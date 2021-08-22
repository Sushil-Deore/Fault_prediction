from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

# Create Flask object to Run
app = Flask(__name__)

# Load the model from the File

model_load = joblib.load('./model/premium_pred_model.pkl')


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
        return render_template('index.html', prediction_text='Insurance Premium is  {}'.format(output))
    else:
        return render_template('index.html')


"""
@app.route("/predict_api", methods=['POST', 'GET'])
def predict_api():
    print(" request.method :", request.method)
    if request.method == 'POST':
        data = request.get_json()
        return jsonify(model_load.predict([np.array(list(data.values()))]).tolist())
    else:
        return render_template('index.html')

"""
if __name__ == '__main__':
    app.run(debug=True, port=5000)
