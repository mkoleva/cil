__author__ = 'karimlabib'

import numpy as np
from sklearn import linear_model, cross_validation, ensemble, svm

def train():

    feature_vector_svd = np.load('data/svd/feature_vector_svd.npy')
    feature_vector_sgd = np.load('data/sgd/feature_vector_sgd_220.npy')
    feature_vector_sgd_c_80 = np.load('data/sgd/feature_vector_sgd_c_80.npy')
    feature_vector_knn = np.load('data/knn/feature_vector_knn.npy')
    feature_vector_knn_items = np.load('data/knn/feature_vector_knn_items.npy')
    feature_vector_rbm = np.load('data/rbm/feature_vector_rbm.npy')

    X, y = [], []

    for i in range(0, np.shape(feature_vector_knn)[0]):
        X.append([feature_vector_svd[i][0],
                  feature_vector_sgd[i][0],
                  feature_vector_sgd_c_80[i][0],
                  feature_vector_knn[i][0],
                  feature_vector_knn_items[i][0],
                  feature_vector_rbm[i][0]])

        y.append(feature_vector_sgd[i][1])

    regressor = linear_model.LinearRegression()
    regressor.fit(X, y)

    scores = cross_validation.cross_val_score(regressor, X, y, scoring="mean_squared_error", cv=10)

    print scores.mean()
    print scores
    print regressor.coef_

    return regressor

def predict(regressor):

    svd = np.genfromtxt('data/svd/my_prediction_svd.csv', delimiter=',', dtype=None)
    sgd = np.genfromtxt('data/sgd/my_prediction_sgd_220.csv', delimiter=',', dtype=None)
    sgd_c_80 = np.genfromtxt('data/sgd/my_prediction_sgd_c_80.csv', delimiter=',', dtype=None)
    knn = np.genfromtxt('data/knn/my_prediction_knn.csv', delimiter=',', dtype=None)
    knn_items = np.genfromtxt('data/knn/my_prediction_knn_items.csv', delimiter=',', dtype=None)
    rbm = np.genfromtxt('data/rbm/my_prediction_rbm.csv', delimiter=',', dtype=None)

    f = open('data/my_prediction_linear_regression.csv', 'w')

    f.write('%s\n' % "Id,Prediction")

    print "prediction"

    for i in range(1, np.shape(sgd)[0]):
        if i % 1000 == 0:
            print i

        entry = sgd[i][0].split("_")
        row = int(entry[0][1:])
        column = int(entry[1][1:])

        svd_rating = float(svd[i][1])
        sgd_rating = float(sgd[i][1])
        sgd_80_rating = float(sgd_c_80[i][1])
        knn_rating = float(knn[i][1])
        knn_items_rating = float(knn_items[i][1])
        rbm_rating = float(rbm[i][1])

        prediction = regressor.predict(np.array([[svd_rating,
                                                  sgd_rating,
                                                  sgd_80_rating,
                                                  knn_rating,
                                                  knn_items_rating,
                                                  rbm_rating]]))

        if prediction > 5:
            prediction = 5

        f.write('r%d_c%d,%f\n' % (row, column, prediction))

    f.close()


def code():
    regressor = train()
    predict(regressor)

if __name__ == '__main__':
    code()