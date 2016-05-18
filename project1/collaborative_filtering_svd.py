import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt

def calculate_error(X_true, X_pred):
    sum = 0
    obs = 0
    for i in range(0, np.shape(X_true)[0]):
        for j in range(0, np.shape(X_true)[1]):
            if X_true[i][j] != 0:
                obs += 1
                sum += (X_true[i][j] - X_pred[i][j]) * (X_true[i][j] - X_pred[i][j])

    return np.sqrt(sum / obs)


def parseInputMatrix(path='data/data_train.csv'):
    """ Reads the data_trains.csv and builds a lists of users, films and ratings """
    input = np.genfromtxt(path, delimiter=',', dtype=None, skip_header=1)

    user, movie, ratings = [], [], []
    for ind, (entry, rating) in enumerate(input):
        # iterate over each row and parse rRowId_cColID, rating
        r,c = entry.split("_")
        row = int(r[1:]) - 1
        column = int(c[1:]) - 1

        user.append(row)
        movie.append(column)
        ratings.append(int(rating))

    training_data = zip(user, movie, ratings)
    return training_data


def computeAverages(matrix):
    """ Calculates mean rating for a film across users and mean rating of user across films.

    TODO: Do values need to be integers?
    """
    numUsers, numFilms = matrix.shape

    # Finds the mean across every column, ignoring 0s and then sets nan to 0
    averages_of_movies = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, matrix)
    averages_of_movies[np.isnan(averages_of_movies)]=0.

    # Finds the mean across every row, ignoring 0s and then sets nan to 0
    averages_of_users = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 1, matrix)
    averages_of_users[np.isnan(averages_of_users)]=0.

    return averages_of_movies, averages_of_users



def code():

    amount_of_validation = 0.01
    # row is user, movie is column
    matrix = np.zeros((10000, 1000))

    training_data = parseInputMatrix()
    np.random.shuffle(training_data)

    counter = 0
    for (u, m, rating) in training_data:
        if counter <= (1 - amount_of_validation) * len(training_data):
            matrix[u][m] = rating
        counter += 1

    averages_of_movies = np.zeros(1000)
    averages_of_users = np.zeros(10000)

    # impute a matrix...
    matrix_imputed = matrix


    print "Compute averages for imputation...\n"

    averages_of_movies, averages_of_users = computeAverages(matrix)
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
                    matrix_imputed[i][j] = 0.5 * averages_of_users[i] + 0.5 * averages_of_movies[j]

    print "finished imputing matrix...\n"

    print "start SVD...\n"

    U, D, V = np.linalg.svd(matrix_imputed, full_matrices=False)

    plt.semilogy(D)
    plt.show()

    print "finished SVD...\n"

    best_K = 1
    rmse_min = 10

    errors_list = []
    for K in range(6, 20):
        D_new = np.zeros((K, K))

        for i in range(0, K):
            D_new[i][i] = np.sqrt(D[i])

        U_new = np.dot(U[:, 0:K], D_new)
        V_new = np.dot(np.transpose(V)[:, 0:K], D_new)

        counter = 0
        error_on_validation = 0
        for (i, j, r) in training_data:

            counter += 1

            if counter <= (1 - amount_of_validation) * len(training_data):
                continue

            error_on_validation += pow(r - np.dot(U_new[i, :], np.transpose(V_new[j, :])), 2)

        rmse_error = np.sqrt(error_on_validation / (amount_of_validation * len(training_data)))

        print K, rmse_error

        errors_list.append(rmse_error)

        if rmse_error < rmse_min:
            rmse_min = rmse_error
            best_K = K

    plt.plot(errors_list)
    plt.show()

    D_new = np.zeros((best_K, best_K))

    for i in range(0, np.shape(D_new)[0]):
        D_new[i][i] = np.sqrt(D[i])

    U_new = np.dot(U[:, 0:best_K], D_new)
    V_new = np.dot(np.transpose(V)[:, 0:best_K], D_new)

    sample_submission = np.genfromtxt('data/sampleSubmission.csv', delimiter=',', dtype=None)

    f = open('data/my_prediction.csv', 'w')

    f.write('%s\n' % "Id,Prediction")

    for i in range(1, np.shape(sample_submission)[0]):
        entry = sample_submission[i][0].split("_")
        row = int(entry[0][1:])
        column = int(entry[1][1:])

        f.write('r%d_c%d,%f\n' % (row, column, np.dot(U_new[row - 1, :], np.transpose(V_new[column - 1, :]))))

    f.close()

if __name__ == '__main__':
    code()
