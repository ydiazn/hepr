import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn import metrics

def multiouput_regressor(input, target, input_test, target_test, output):
    # dataset
    X = input
    y = target
    X_test = input_test
    y_test = target_test

    estimator = LinearRegression()
    model = MultiOutputRegressor(estimator)

    # Perform 6-fold cross validation
    #scores = cross_val_score(model, X, y, cv=5)
    #print("Cross-validated scores: ")
    #print(scores)

    # Make cross validated predictions
    scores = cross_validate(model, X, y, cv=5, return_estimator=True)
    model2 = scores['estimator'][1]

    predictions = model2.predict(X_test)

    # Remove exterme values
    mask = predictions[:, 1] <= 1
    y_test = y_test[mask]
    predictions = predictions[mask]

    accuracy = metrics.r2_score(y_test, predictions)
    print("Cross-Predicted Accuracy: {}".format(accuracy))

    # The line / model
    fig, ax = plt.subplots()
    ax.scatter(y_test[:, 0], y_test[:, 1], color='red', alpha=0.5)
    ax.scatter(predictions[:, 0], predictions[:, 1], color='blue', alpha=0.5)
    ax.set_xlabel('P')
    ax.set_ylabel('Q')
    plt.show()

    np.savetxt(output, predictions)
