"""
Gradient descent algorithm for multiple linear regression

Implement the gradient descent algorithm
"""
from __future__ import print_function
import numpy as np
import pandas
# pandas data type dict
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
              'sqft_living15':float, 'grade':int, 'yr_renovated':int,
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
              'sqft_lot15':float, 'sqft_living':float, 'floors':str,
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}


def get_features_output(dataset, features, output):
    """
    dataset: pandas data frame
    features: list of feature names (strings)
    output: string of output name
    """
    features_matrix = dataset.as_matrix(features)
    constant = np.ones(len(features_matrix))
    features_matrix = np.column_stack((constant, features_matrix))

    output_array = np.squeeze(dataset.as_matrix([output]))

    return (features_matrix, output_array)


def predict_output(features_matrix, weights):
    """
    returns a prediction
    """
    predictions = np.dot(features_matrix, weights)
    return predictions

def feature_derivative(errors, feature):
    """
    return derivative w.r.t to given feature for gradient descent

    errors: vector of (prediction - output) values
    feature: vector of `feature` values for each training data point
    """
    derivative = 2 * np.dot(errors, feature)
    return derivative

def regression_gradient_descent(features_matrix, output_array, initial_weights,
    step_size, tolerance):

    gradient = tolerance + 1 # just some random initialization
    weights = np.array(initial_weights)

    # update weights based on gradient descent
    while gradient > tolerance:
        predictions = predict_output(features_matrix, weights)
        errors = predictions - output_array

        # update each weight inidividually
        gradient_sum_squares = 0
        for i in range(len(weights)):
            feature = features_matrix[:, i] # feature column
            derivative = feature_derivative(errors, feature)

            # euclidean sum of gradient wrt to each feature weight
            gradient_sum_squares += derivative ** 2

            # update weights
            weights[i] = weights[i] - (step_size * derivative)

        gradient = np.sqrt(gradient_sum_squares)

    return weights


def main():
    """
    use gradient descent here
    """
    # training data
    train = pandas.read_csv('kc_house_train_data.csv', dtype=dtype_dict)

    # test data
    test = pandas.read_csv('kc_house_test_data.csv', dtype=dtype_dict)

    
    simple_features = ['sqft_living']
    _output = 'price'
    simple_feature_matrix, output = get_features_output(train, simple_features,
                                                        _output)
    initial_weights = np.array([-47000., 1.])
    step_size = 7e-12
    tolerance = 2.5e7
    simple_weights = regression_gradient_descent(simple_feature_matrix, output,
                                                 initial_weights, step_size,
                                                 tolerance)

   # weight for sqft_living
    print(np.round(simple_weights[1], 1))

    # construct test data input and output matrices
    test_feature_matrix, test_output = get_features_output(test, simple_features, _output)
    # predict prices for test data
    test_predictions = predict_output(test_feature_matrix, simple_weights)

    # predicted price for the 1st house in the TEST data set for model 1
    print(np.round(test_predictions[0], 0)) 

    test_errors = test_output - test_predictions
    rss = np.sum(test_errors ** 2)
    print(rss)

    # second model
    model_features = ['sqft_living', 'sqft_living15']
    _output = 'price'
    initial_weights = np.array([-100000., 1., 1.])
    step_size = 4e-12
    tolerance = 1e9
    feature_matrix, output = get_features_output(train, model_features,
                                                        _output)
    model_weights = regression_gradient_descent(feature_matrix, output,
                                                initial_weights, step_size,
                                                tolerance)
    test_feature_matrix, test_output = get_features_output(test,
                                                           model_features,
                                                           _output)

    # predict with 2nd model
    new_predictions = predict_output(test_feature_matrix, model_weights)

    # predicted price for the 1st house in the TEST data set for model 2
    print(np.round(new_predictions[0], 0))

    # actual price of 1st house
    print(test_output[0])

    # rss on second model
    test_errors = test_output - new_predictions
    rss = np.sum(test_errors ** 2)
    print(rss)


if __name__ == '__main__':
    main()
