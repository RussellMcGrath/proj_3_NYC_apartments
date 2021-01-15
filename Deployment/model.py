import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('streeteasy.csv')

# dataset['rate'].fillna(0, inplace=True)

# dataset['sales_in_first_month'].fillna(dataset['sales_in_first_month'].mean(), inplace=True)

# X = dataset.iloc[:, :3]

# def convert_to_int(word):
#     word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
#                 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
#     return word_dict[word]

# X['rate'] = X['rate'].apply(lambda x : convert_to_int(x))

# y = dataset.iloc[:, -1]

X=dataset[['bedrooms','bathrooms','size_sqft','min_to_subway','floor','building_age_yrs','no_fee','has_roofdeck']]

#'has_washer_dryer','has_doorman','has_elevator','has_dishwasher','has_patio','has_gym']]

y=dataset['rent']

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X, y)

pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 1, 700, 10, 1,20,1,1]]))