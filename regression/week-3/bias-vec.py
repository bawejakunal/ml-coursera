"""
Gradient descent algorithm for multiple linear regression

Implement the gradient descent algorithm

vectorize gradient calculation
"""
from __future__ import print_function
import numpy as np
from sklearn import linear_model
import pandas

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# pandas data type dict
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
              'sqft_living15':float, 'grade':int, 'yr_renovated':int,
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
              'sqft_lot15':float, 'sqft_living':float, 'floors':str,
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

def polynomial_dframe(feature, degree):
    """
    feature: array of feature values for given data
    degree: polynomial degree upto which columns are added to data frame
    """

    # create a data frame with first column equal to feature
    dframe = pandas.DataFrame({'power_1': feature})

    for power in range(2, degree+1):
        name = 'power_%d' % power
        dframe[name] = dframe['power_1'].apply(lambda x: x**power)
    return dframe

def estimate_poly(data, degree, fignum=0):
    poly_data = polynomial_dframe(data['sqft_living'], degree)
    features = poly_data.as_matrix()
    model = linear_model.LinearRegression()
    model.fit(features, data['price'])

    if fignum > 0:
        plt.figure(fignum)
        plt.plot(poly_data['power_1'], data['price'], '.',
            poly_data['power_1'], model.predict(features), '-')
        plt.savefig('fig-%d' % fignum)

    return model

def main():
    sales = pandas.read_csv('kc_house_data.csv', dtype=dtype_dict)
    sales = sales.sort_values(by=['sqft_living', 'price'])

    # on original data set
    model1 = estimate_poly(sales, 1, 1)
    model2 = estimate_poly(sales, 2, 2)
    model3 = estimate_poly(sales, 3, 3)
    model15 = estimate_poly(sales, 15, 4)

    # set 1
    sales_1 = pandas.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
    sales_1 = sales_1.sort_values(by=['sqft_living', 'price'])
    model_set_1 = estimate_poly(sales_1, 15, 5)
    print(model_set_1.coef_)

    # set 2
    sales_2 = pandas.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
    sales_2 = sales_2.sort_values(by=['sqft_living', 'price'])
    model_set_2 = estimate_poly(sales_2, 15, 6)
    print(model_set_2.coef_)

    # set 3
    sales_3 = pandas.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
    sales_3 = sales_3.sort_values(by=['sqft_living', 'price'])
    model_set_3 = estimate_poly(sales_3, 15, 7)
    print(model_set_3.coef_)

    # set 4
    sales_4 = pandas.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
    sales_4 = sales_4.sort_values(by=['sqft_living', 'price'])
    model_set_4 = estimate_poly(sales_4, 15, 8)
    print(model_set_4.coef_)

    # for degrees 1 to 15 for train, validate, test data
    train = pandas.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
    valid = pandas.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)
    test = pandas.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)

    min_deg = None
    minimum = None
    models = list()
    for degree in range(1, 16):
        model = estimate_poly(train, degree)
        models.append(model)
        features = polynomial_dframe(valid['sqft_living'], degree).as_matrix()
        predict = model.predict(features)
        errors = predict - valid['price']
        rss = np.sum(errors**2)
        if minimum is None or rss < minimum:
            minimum = rss
            min_deg = degree

    # rss on test data
    features = polynomial_dframe(test['sqft_living'], min_deg)
    predict = models[min_deg-1].predict(features)
    errors = predict - test['price']
    rss = np.sum(errors**2)
    print(rss)

if __name__ == '__main__':
    main()
