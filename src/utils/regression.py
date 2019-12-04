import numpy as np
import matplotlib.pyplot as plt
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

    # Create a random dataset
    # rng = np.random.RandomState(1)
    # X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
    # y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    # y += (0.5 - rng.rand(*y.shape))
    #train_size = 150

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=4)

    modell = Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('linear', LinearRegression(fit_intercept=False))
    ])

    model = MultiOutputRegressor(modell)

    # Perform 6-fold cross validation
    #scores = cross_val_score(model, X, y, cv=5)
    #print("Cross-validated scores: ")
    #print(scores)

    # Make cross validated predictions
    scores = cross_validate(model, X, y, cv=5, return_estimator=True)
    model2 = scores['estimator'][1]

    model.fit(X, y)

    predictions = model2.predict(X_test)
    accuracy = metrics.r2_score(y_test, predictions)
    print("Cross-Predicted Accuracy: {}".format(accuracy))

    # The line / model
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

    # Predict on new data
    # fig, ax = plt.subplots()
    # ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    # ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    # ax.set_xlabel('Measured')
    # ax.set_ylabel('Predicted')
    # plt.show()

    np.savetxt(output, predictions)
