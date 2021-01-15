import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sklearn


app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))
limits = [450.0, 1800.0]
model_dict = {
    "manhattan":pickle.load(open('../models/Manhattan_model_2.pkl', 'rb')),
    "queens":pickle.load(open('../models/Queens_model_2.pkl', 'rb')),
    "brooklyn":pickle.load(open('../models/Brooklyn_model_2.pkl', 'rb'))
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [x for x in request.form.values()]
    model = model_dict[int_features[0].lower()]
    int_features = [float(x) for x in int_features[1:]]
    final_features = [np.array(int_features)]
    final_features[0][0] = min(max(final_features[0][0], limits[0]), limits[1])
    prediction = model.predict(final_features)

    
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Rent should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    int_features = [x for x in request.form.values()]
    model = model_dict[int_features[0]]
    int_features = [float(x) for x in int_features[1:3]]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)