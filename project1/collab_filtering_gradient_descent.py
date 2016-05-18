__author__ = 'karimlabib'

import numpy as np
import random
import math


def matrix_factorization(user, movie, rating, P, Q, K, steps=30, alpha=0.005, beta=0.001):

    percentage_training = 0.2
    Q = Q.T

    split_index = int(percentage_training * len(user))
    user_validation = user[split_index:]
    movie_validation = movie[split_index:]
    rating_validation = rating[split_index:]

    validation_data = zip(user_validation, movie_validation, rating_validation)

    user_training = user[:split_index]
    movie_training = movie[:split_index]
    rating_training = rating[:split_index]

    print split_index

    for step in xrange(steps):
        print step
        counter = 0

        training_data = zip(user_training, movie_training, rating_training)
        random.shuffle(training_data)

        for (i, j, r) in training_data:
            if counter % 10000 == 0:
                print counter
            counter += 1

            eij = r - np.dot(P[i, :], Q[:, j])

            eta = alpha

            nP = P[i, :]
            for k in xrange(K):
                nP[k] = P[i][k] - 2 * eta * (eij * Q[k][j] - beta * P[i][k])
                Q[k][j] = Q[k][j] - 2 * eta * (eij * P[i][k] - beta * Q[k][j])

            P[i, :] = nP

        e = 0
        counter = 0

        for (i, j, r) in validation_data:
            counter += 1
            e += pow(r - np.dot(P[i, :], Q[:, j]), 2)

        print np.sqrt(e / (int((1-percentage_training) * len(user))))

    return P, Q.T


def code():

    training_data = np.genfromtxt('data_train_2.csv', delimiter=',', dtype=None)

    # row is user, movie is column

    user, movie, rating = [], [], []
    for i in range(1, np.shape(training_data)[0]):
        entry = training_data[i][0].split("_")
        row = int(entry[0][1:]) - 1
        column = int(entry[1][1:]) - 1

        user.append(row)
        movie.append(column)
        rating.append(int(training_data[i][1]))

    n = 10000
    m = 1000
    k = 40

    P = np.random.normal(0, 0.01, (n, k))
    Q = np.random.normal(0, 0.01, (m, k))

    #P = np.load('nP_matrix.npy')
    #Q = np.load('nQ_matrix.npy')

    nP, nQ = matrix_factorization(user, movie, rating, P, Q, k)
    nR = np.dot(nP, nQ.T)

    np.save('nP_matrix', nP)
    np.save('nQ_matrix', nQ)

    sample_submission = np.genfromtxt('sampleSubmission_new.csv', delimiter=',', dtype=None)

    f = open('my_prediction.csv', 'w')

    f.write('%s\n' % "Id,Prediction")

    for i in range(1, np.shape(sample_submission)[0]):
        entry = sample_submission[i][0].split("_")
        row = int(entry[0][1:])
        column = int(entry[1][1:])

        if nR[row - 1][column - 1] > 5:
            f.write('r%d_c%d,%f\n' % (row, column, 5))
        else:
            f.write('r%d_c%d,%f\n' % (row, column, nR[row - 1][column - 1]))

    f.close()


if __name__ == '__main__':
    code()
