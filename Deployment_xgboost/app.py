# deploying xgboost model

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sklearn
import sys

# import custom_modeler function
from Deployment_xgboost.custom_modeler import custom_modeler
# import neighborhood lists
from Deployment_xgboost.dropdownlists import boroughs, neighborhoods, submarkets

### CREATE FLASK APP ###
app = Flask(__name__)

# import ML models
model = pickle.load(open('Deployment_xgboost/XGBModel.pkl', 'rb'))
sc = pickle.load(open('Deployment_xgboost/X_scaler.pkl','rb'))
y_sc = pickle.load(open('Deployment_xgboost/y_scaler.pkl','rb'))

model_columns = 'Deployment_xgboost/XGB_model_columns.pkl'

BOROUGH_DICT = {
    'brooklyn': [1,0,0],
    'manhattan': [0,1,0],
    'queens': [0,0,1]
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/optimal')
def optimal():
    return render_template('optimal.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [x for x in request.form.values()]
    int_features = [float(x) for x in int_features[1:]]
    final_features = [np.array(int_features + BOROUGH_DICT[(request.form['borough']).lower()])]
    final_features_scaled = sc.transform(final_features)		
    prediction = model.predict(final_features_scaled)
    
    output = round(y_sc.inverse_transform(prediction)[0], 0)

    return render_template('optimal.html', prediction_text='Predicted Rent: ${}'.format(output))

@app.route('/custom')
def custom():
    boroughs_dict = {"list": boroughs}
    neighborhoods_dict = {"list": neighborhoods}
    submarkets_dict = {"list": submarkets}
    return render_template('custom.html', boroughs=boroughs_dict, neighborhoods=neighborhoods_dict, submarkets=submarkets_dict)

@app.route('/custom_results', methods=['POST'])
def custom_results():
    # capture user inputs
    feature_values = [x for x in request.form.values()]
    feature_keys = [x for x in request.form.keys()]
    user_input = {}
    for i in range(len(feature_keys)):
        user_input[feature_keys[i]] = feature_values[i]
    
    # run inputs through custom modeler function
    results = custom_modeler(user_input)
    print(results)
    # extract results of each model
    lm = results["lm"]
    lasso = results["lasso"]
    ridge = results["ridge"]
    elasticnet = results["elas"]
    xgboost = results["xgb"]
    # reformat dropdown list variable to dictionaries so they can be passed to javascript
    boroughs_dict = {"list": boroughs}
    neighborhoods_dict = {"list": neighborhoods}
    submarkets_dict = {"list": submarkets}
    # render page and send variables
    return render_template('custom.html', results_matrix=results, boroughs=boroughs_dict, neighborhoods=neighborhoods_dict, submarkets=submarkets_dict)

# data visualizations endpoint
@app.route('/tableau')
def tableau():
    return render_template('tableau.html')

if __name__ == "__main__":
    app.run(debug=True)