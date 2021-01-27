def custom_modeler(user_input):
    # Set the seed value for the notebook so the results are reproducible
    from numpy.random import seed
    seed(1)
    # import tensorflow
    import pandas as pd
    import numpy as np

    #import data csv
    df = pd.read_csv('input_data/streeteasy.csv')

    # remove extra columns
    trimmed_df = df.drop(columns=["rental_id","building_id"])
    # convert categorical data into dummy columns
    dummied_df = pd.get_dummies(df)
    
    # convert user input values to dummified format
    new_input_dict = {}
    boroughs = sorted(list(df["borough"].unique()))
    neighborhoods = sorted(list(df["neighborhood"].unique()))
    submarkets = sorted(list(df["submarket"].unique()))

    for feature in user_input:
        if feature == "borough":            
            for borough in boroughs:
                if user_input["borough"] == borough:
                    new_input_dict[f"borough_{borough}"] = 1
                else:
                    new_input_dict[f"borough_{borough}"] = 0
        elif feature == "neighborhood":            
            for neighborhood in neighborhoods:
                if user_input["neighborhood"] == neighborhood:
                    new_input_dict[f"neighborhood_{neighborhood}"] = 1
                else:
                    new_input_dict[f"neighborhood_{neighborhood}"] = 0
        elif feature == "submarket":
            for submarket in submarkets:
                if user_input["submarket"] == submarket:
                    new_input_dict[f"submarket_{submarket}"] = 1
                else:
                    new_input_dict[f"submarket_{submarket}"] = 0
        else:
            new_input_dict[feature] = user_input[feature]

    # use user input selection to create selected features dataframe
    selected_features = list(new_input_dict.keys())
    selection_df = dummied_df[selected_features]

    # Assign X (data) and y (target)
    X = selection_df
    y = df["rent"].values.reshape(-1, 1)
    
    # Split the data into training and testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    ### SCALE DATA ###
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    # from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import StandardScaler
    # Create a StandardScater model and fit it to the training data
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    # Transform the training and testing data using the X_scaler and y_scaler models
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    ##### CREATE MODELS #####
    # model error dictionary
    all_models = {"models":[],
             "mse":[],
             "r2":[]}
    
    ### LINEAR REGRESSION ###
    from sklearn.linear_model import LinearRegression 
    lm_model = LinearRegression()
    lm_model.fit(X_train_scaled, y_train_scaled)
    lm_predictions = lm_model.predict(X_train_scaled)
    from sklearn.metrics import mean_squared_error
    lm_pred= lm_model.predict(X_test_scaled)
    from sklearn.metrics import mean_squared_error
    MSE = mean_squared_error(y_train_scaled, lm_predictions)
    r2 = lm_model.score(X_test_scaled, y_test_scaled)
    # save error results
    all_models["models"].append(lm_model)
    all_models["mse"].append(MSE)
    all_models["r2"].append(r2)

    ### LASSO ###
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=.01).fit(X_train_scaled, y_train_scaled)
    lasso_predictions = lasso.predict(X_test_scaled)
    MSE = mean_squared_error(y_test_scaled, lasso_predictions)
    r2 = lasso.score(X_test_scaled, y_test_scaled)
    # save error results
    all_models["models"].append(lasso)
    all_models["mse"].append(MSE)
    all_models["r2"].append(r2)

    ### RIDGE ###
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=.01).fit(X_train_scaled, y_train_scaled)
    ridge_predictions = ridge.predict(X_test_scaled)
    MSE = mean_squared_error(y_test_scaled, ridge_predictions)
    r2 = ridge.score(X_test_scaled, y_test_scaled)
    # save error results
    all_models["models"].append(ridge)
    all_models["mse"].append(MSE)
    all_models["r2"].append(r2)

    ### ELASTICNET ###
    from sklearn.linear_model import ElasticNet
    elasticnet = ElasticNet(alpha=.01).fit(X_train_scaled, y_train_scaled)
    el_predictions = elasticnet.predict(X_test_scaled)
    MSE = mean_squared_error(y_test_scaled, el_predictions)
    r2 = elasticnet.score(X_test_scaled, y_test_scaled)
    # save error results
    all_models["models"].append(elasticnet)
    all_models["mse"].append(MSE)
    all_models["r2"].append(r2)

    ### XGBOOST REGRESSOR ###
    from xgboost import XGBRegressor
    XGBModel = XGBRegressor()
    XGBModel.fit(X_train_scaled, y_train_scaled , verbose=False)
    XGBpredictions = XGBModel.predict(X_test_scaled)
    MSE = mean_squared_error(y_test_scaled, XGBpredictions)
    r2 = XGBModel.score(X_test_scaled, y_test_scaled)
    # save error results
    all_models["models"].append(XGBModel)
    all_models["mse"].append(MSE)
    all_models["r2"].append(r2)

    ### Neural Network ###
    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Dense
    # # Define model
    # model = Sequential()
    # model.add(Dense(500, input_dim=len(new_input_dict.values()), activation= "relu"))
    # model.add(Dense(100, activation= "relu"))
    # model.add(Dense(50, activation= "relu"))
    # model.add(Dense(1))
    # # Compile and fit the model
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    # # Fit the model to the training data
    # model.fit(
    #     X_train_scaled,
    #     y_train_scaled,
    #     epochs=100,
    #     shuffle=True,
    #     verbose=2
    # )
    # pred_train_scaled= model.predict(X_train_scaled)
    # MSE = np.sqrt(mean_squared_error(y_train_scaled,pred_train_scaled))
    # pred = model.predict(X_test_scaled)
    # r2 = np.sqrt(mean_squared_error(y_test_scaled, pred))
    # # save error results
    # all_models["models"].append(model)
    # all_models["mse"].append(MSE)
    # all_models["r2"].append(r2)

    #### GET PREDICTIONS ####
    # scale input values
    input_values = list(new_input_dict.values())
    input_values_scaled = X_scaler.transform(pd.DataFrame(new_input_dict, index=[0]))
    # enter values into model, and inverse tranform the results
    # Linear Regression Model
    lm_pred_array = y_scaler.inverse_transform([lm_model.predict(input_values_scaled),])
    lm_pred = lm_pred_array[0][0][0]
    # Lasso Model
    lasso_pred_array = y_scaler.inverse_transform([lasso.predict(input_values_scaled),])
    lasso_pred = lasso_pred_array[0][0]
    # Ridge Model
    ridge_pred_array = y_scaler.inverse_transform([ridge.predict(input_values_scaled),])
    ridge_pred = ridge_pred_array[0][0][0]
    # ElasticNet Model
    elas_pred_array = y_scaler.inverse_transform([elasticnet.predict(input_values_scaled),])
    elasticnet_pred = elas_pred_array[0][0]
    # XGBoost Regressor Model
    XGB_pred_array = y_scaler.inverse_transform([XGBModel.predict(input_values_scaled),])
    XGB_pred = XGB_pred_array[0][0]
    # Neural Network Model
    # nn_pred_array = y_scaler.inverse_transform([model.predict(input_values_scaled),])
    # nn_pred = nn_pred_array[0][0][0]

    # assemble results into dictionary
    results = {
        "lm": {"model":"Linear Regression",
            "r2": round(all_models["r2"][0],4),
            "prediction": round(lm_pred,2)},
        "lasso": {"model":"Lasso",
            "r2": round(all_models["r2"][1],4),
            "prediction": round(lasso_pred,2)},
        "ridge": {"model":"Ridge",
            "r2": round(all_models["r2"][2],4),
            "prediction": round(ridge_pred,2)},
        "elas": {"model":"ElasticNet",
            "r2": round(all_models["r2"][3],4),
            "prediction": round(elasticnet_pred,2)},
        "xgb": {"model":"XGBoost Regressor",
            "r2": round(all_models["r2"][4],4),
            "prediction": round(XGB_pred,2)}#,
        # "nn": {"model":"Neural Network",
        #     "r2": round(all_models["r2"][5],4),
        #     "prediction": round(nn_pred,2)},
    }

    return (results,boroughs,neighborhoods,submarkets)
