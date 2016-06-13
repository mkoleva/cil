__author__ = 'karimlabib'

import numpy as np

from general_functions import *
from sklearn.metrics.pairwise import pairwise_distances


numRatings = 0
valSetSize = 0
amount_of_validation = 0.01
"""this is the pearson distance global weight matrix"""
weight_matrix = np.zeros((10000, 10000))
averages_of_movies, averages_of_users = [], []
rating_matrix = np.zeros((10000, 1000))

movies_indices_rated_by_user = [set() for _ in xrange(10000)]


def predict_rating(user, movie):

    # indices_of_nearest_neigbours = get_indices_of_nearest_neighbors(weight_matrix[user, :], k)
    # nearestNeighbours = np.shape(indices_of_nearest_neigbours)[0]

    numerator, denominator = 0.0, 0.0

    this_user = rating_matrix[user]
    rated_films = np.nonzero(this_user)[0]
    # print "Rated films ", len(rated_films)
    for movieInd in rated_films:

        if this_user[movieInd] != 0:
            numerator += weight_matrix[movieInd, movie]*this_user[movieInd]

        denominator += weight_matrix[movieInd, movie]

    """
    rating_nearest_neigbours_normalized = np.subtract(np.take(rating_matrix[:,movie], indices_of_nearest_neigbours), np.take(averages_of_users, indices_of_nearest_neigbours))

    numerator = sum(np.multiply(np.take(weight_matrix[user, :], indices_of_nearest_neigbours), rating_nearest_neigbours_normalized))

    denominator = sum(np.take(weight_matrix[user, :], indices_of_nearest_neigbours))

    """
    predicted_rating = numerator / denominator

    return predicted_rating

def compute_weight_matrix_users():
    print "total ratings to iterate: ", np.shape(rating_matrix)[0]
    # pool = multiprocessing.Pool(4)
    # out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))
    weight_matrix = pairwise_distances(rating_matrix.T, metric='correlation', n_jobs=-1) #user similarity
    weight_matrix = np.nan_to_num(weight_matrix)


    # for i in range(0, np.shape(rating_matrix)[0]):
    #     print i

    #     for j in range(i + 1, np.shape(rating_matrix)[0]):

    #         weight_matrix[i][j] = weight_matrix[j][i] = pearson_distance(i, j)

    np.save('data/knn/weight_matrix_items', weight_matrix)

def do_prediction():
    sample_submission = np.genfromtxt('data/sampleSubmission.csv', delimiter=',', dtype=None)

    f = open('data/knn/my_prediction_knn_items.csv', 'w')
    f.write('%s\n' % "Id,Prediction")

    for i in range(1, np.shape(sample_submission)[0]):
        if i % 1000 == 0:
            print i

        entry = sample_submission[i][0].split("_")
        user = int(entry[0][1:])
        movie = int(entry[1][1:])

        predicted_rating = predict_rating(user - 1, movie - 1)

        if predicted_rating > 5:
            f.write('r%d_c%d,%f\n' % (user, movie, 5))
        else:
            f.write('r%d_c%d,%f\n' % (user, movie, predicted_rating))

    f.close()

def do_validation(validationSubset):

    error = 0

    for (u, m, rating) in validationSubset:
        predicted_rating = predict_rating(u, m)
        # print predicted_rating

        error += pow((predicted_rating - rating), 2)

    error /= np.shape(validationSubset)[0]

    print np.sqrt(error)

    feature_vector_for_regression = []

    for (u, m, rating) in validationSubset:
        predicted_rating = predict_rating(u, m)
        feature_vector_for_regression.append([predicted_rating, rating])

    np.save('data/knn/feature_vector_knn_items', feature_vector_for_regression)

def code():

    weight_matrix_and_validation_indices_computed = True

    global weight_matrix, averages_of_users, averages_of_movies, valSetSize, numRatings

    trainingSubset = parseInputMatrix()

    """ Partitioning the training data into training and validation"""

    indices_for_validation = generate_validation_set(trainingSubset, indices_for_validation_set_already_chosen=weight_matrix_and_validation_indices_computed)
    print indices_for_validation[:10]
    validationSubset = np.take(trainingSubset, indices_for_validation, axis=0)
    trainingSubset = np.delete(trainingSubset, indices_for_validation, axis=0)


    print("shape of training set: ", trainingSubset.shape)
    print("shape of validation set: ", validationSubset.shape)


    for (u, m, rating) in trainingSubset:
        movies_indices_rated_by_user[u].add(m)
        rating_matrix[u][m] = rating

    if weight_matrix_and_validation_indices_computed == True:
        try:
            weight_matrix = np.load('data/knn/weight_matrix_items.npy')
        except Exception as e:
            print "Coundn't load the matrix ", e
            compute_weight_matrix_users()
    else:
        compute_weight_matrix_users()


    do_validation(validationSubset)

    do_prediction()


if __name__ == '__main__':
    code()