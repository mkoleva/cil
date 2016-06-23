import numpy as np
import matplotlib.pyplot as plt

from general_functions import *

rating_matrix = []

# SVD output
U, D, V = [], [], []

def calculate_error(X_true, X_pred):
    sum = 0
    obs = 0
    for i in range(0, np.shape(X_true)[0]):
        for j in range(0, np.shape(X_true)[1]):
            if X_true[i][j] != 0:
                obs += 1
                sum += (X_true[i][j] - X_pred[i][j]) * (X_true[i][j] - X_pred[i][j])

    return np.sqrt(sum / obs)


def impute_rating_matrix():
    # impute a matrix...
    matrix_imputed = rating_matrix


    print "Compute averages for imputation...\n"

    averages_of_movies, averages_of_users = computeAverages(rating_matrix)
    print "Finished computing averages for imputation...\n"

    print "Started imputing matrix...\n"

    for i in range(0, 10000):
        for j in range(0, 1000):
            if matrix_imputed[i][j] == 0:
                if averages_of_users[i] == 0:
                    matrix_imputed[i][j] = averages_of_movies[j]
                elif averages_of_movies[j] == 0:
                    matrix_imputed[i][j] = averages_of_users[j]
                else:
                    matrix_imputed[i][j] = 0.25 * averages_of_users[i] + 0.75 * averages_of_movies[j]

    print "finished imputing matrix...\n"

    return matrix_imputed

def do_validation(validationSubset):
    best_K = 1
    rmse_min = 10

    for K in range(6, 20):
        D_new = np.zeros((K, K))

        for i in range(0, K):
            D_new[i][i] = np.sqrt(D[i])

        U_new = np.dot(U[:, 0:K], D_new)
        V_new = np.dot(np.transpose(V)[:, 0:K], D_new)

        counter = 0
        error_on_validation = 0
        for (i, j, r) in validationSubset:

            counter += 1

            error_on_validation += pow(r - np.dot(U_new[i, :], np.transpose(V_new[j, :])), 2)

        rmse_error = np.sqrt(error_on_validation / np.shape(validationSubset)[0])

        print K, rmse_error

        if rmse_error < rmse_min:
            rmse_min = rmse_error
            best_K = K

    print "best K = ", best_K

    # save for regression
    D_new = np.zeros((best_K, best_K))

    for i in range(0, best_K):
        D_new[i][i] = np.sqrt(D[i])

    U_new = np.dot(U[:, 0:best_K], D_new)
    V_new = np.dot(np.transpose(V)[:, 0:best_K], D_new)

    feature_vector_for_regression = []

    for (i, j, r) in validationSubset:
        prediction = np.dot(U_new[i, :], np.transpose(V_new[j, :]))
        feature_vector_for_regression.append([prediction, r])

    np.save('data/svd/feature_vector_svd', feature_vector_for_regression)


    return best_K

def do_prediction(best_K=12):
    D_new = np.zeros((best_K, best_K))

    for i in range(0, np.shape(D_new)[0]):
        D_new[i][i] = np.sqrt(D[i])

    U_new = np.dot(U[:, 0:best_K], D_new)
    V_new = np.dot(np.transpose(V)[:, 0:best_K], D_new)

    np.save('data/svd/nP_matrix_svd', U_new)
    np.save('data/svd/nQ_matrix_svd', V_new)

    sample_submission = np.genfromtxt('data/sampleSubmission.csv', delimiter=',', dtype=None)

    f = open('data/svd/my_prediction_svd.csv', 'w')

    f.write('%s\n' % "Id,Prediction")

    for i in range(1, np.shape(sample_submission)[0]):
        entry = sample_submission[i][0].split("_")
        row = int(entry[0][1:])
        column = int(entry[1][1:])

        f.write('r%d_c%d,%f\n' % (row, column, np.dot(U_new[row - 1, :], np.transpose(V_new[column - 1, :]))))

    f.close()

def code():

    global rating_matrix, U, D, V

    # row is user, movie is column
    rating_matrix = np.zeros((10000, 1000))

    training_data = parseInputMatrix()
    """ Partitioning the training data into training and validation"""

    indices_for_validation = generate_validation_set(training_data, indices_for_validation_set_already_chosen=True)
    print indices_for_validation[:10]

    validationSubset = np.take(training_data, indices_for_validation, axis=0)
    trainingSubset = np.delete(training_data, indices_for_validation, axis=0)


    counter = 0
    for (u, m, rating) in trainingSubset:
        rating_matrix[u][m] = rating

    averages_of_movies = np.zeros(1000)
    averages_of_users = np.zeros(10000)


    matrix_imputed = impute_rating_matrix()

    print "start SVD...\n"

    U, D, V = np.linalg.svd(matrix_imputed, full_matrices=False)

    print "finished SVD...\n"

    best_K = do_validation(validationSubset)

    do_prediction(best_K)

if __name__ == '__main__':
    code()
