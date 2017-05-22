"""
Gradient descent algorithm for multiple linear regression
"""
from __future__ import print_function
import numpy as np
import pandas
from sklearn import linear_model

# pandas data type dict
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
              'sqft_living15':float, 'grade':int, 'yr_renovated':int,
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
              'sqft_lot15':float, 'sqft_living':float, 'floors':str,
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

# training data
train = pandas.read_csv('kc_house_train_data.csv', dtype=dtype_dict)

# test data
test = pandas.read_csv('kc_house_test_data.csv', dtype=dtype_dict)

# add new columns
train['bedrooms_squared'] = train['bedrooms'] ** 2
train['bed_bath_rooms'] = train['bedrooms'] * train['bathrooms']
train['log_sqft_living'] = np.log(train['sqft_living'])
train['lat_plus_long'] = train['lat'] + train['long']

test['bed_bath_rooms'] = test['bedrooms'] * test['bathrooms']
test['bedrooms_squared'] = test['bedrooms'] ** 2
test['log_sqft_living'] = np.log(test['sqft_living'])
test['lat_plus_long'] = test['lat'] + test['long']

# average on test data
print(np.round(test['bedrooms_squared'].mean(), 2))
print(np.round(test['bed_bath_rooms'].mean(), 2))
print(np.round(test['log_sqft_living'].mean(), 2))
print(np.round(test['lat_plus_long'].mean(), 2))


# construct linear models
length = len(train)

# price is common for all models
price = train['price'].as_matrix().reshape(length, 1)

# Model 1: 'sqft_living', 'bedrooms', 'bathrooms', 'lat', and 'long'
features_1 = train.as_matrix(['sqft_living', 'bedrooms', 'bathrooms', 'lat',
                              'long'])
model_1 = linear_model.LinearRegression().fit(features_1, price)
# coefficients
print(model_1.coef_)


# Model 2: 'sqft_living', 'bedrooms', 'bathrooms', 'lat','long', and
# 'bed_bath_rooms'
features_2 = train.as_matrix(['sqft_living', 'bedrooms', 'bathrooms', 'lat','long', 'bed_bath_rooms'])
model_2 = linear_model.LinearRegression().fit(features_2, price)
# coefficients
print(model_2.coef_)


# Model 3: 'sqft_living', 'bedrooms', 'bathrooms', 'lat','long',
# 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', and 'lat_plus_long'
features_3 = train.as_matrix(['sqft_living', 'bedrooms', 'bathrooms', 'lat',
                              'long', 'bed_bath_rooms', 'bedrooms_squared',
                              'log_sqft_living', 'lat_plus_long'])
model_3 = linear_model.LinearRegression().fit(features_3, price)
# coefficients
print(model_3.coef_)


# residual sum of squares on training data
print(model_1.residues_)
print(model_2.residues_)
print(model_3.residues_)

# mean squared error on test data
test_len = len(test)
error_1 = np.mean((model_1.predict(test.as_matrix(['sqft_living', 'bedrooms',
                                                   'bathrooms', 'lat', 'long'])
                                  ) - test['price'].values.reshape(test_len, 1)
                  )**2)
print(error_1)

# mean squared error for second model on test data
error_2 = np.mean((model_2.predict(test.as_matrix(['sqft_living', 'bedrooms',
                                                   'bathrooms', 'lat', 'long',
                                                   'bed_bath_rooms']))\
        - test['price'].values.reshape(test_len, 1))**2)
print(error_2)

# mean squared error for third model
error_3 = np.mean((model_3.predict(test.as_matrix(['sqft_living', 'bedrooms',
                                                   'bathrooms', 'lat', 'long',
                                                   'bed_bath_rooms',
                                                   'bedrooms_squared',
                                                   'log_sqft_living',
                                                   'lat_plus_long']))\
        - test['price'].values.reshape(test_len, 1))**2)
print(error_3)
