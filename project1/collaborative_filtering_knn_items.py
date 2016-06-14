__author__ = 'karimlabib'

import numpy as np

from general_functions import *
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, cdist


numRatings = 0
valSetSize = 0
amount_of_validation = 0.01
# this is the pearson distance global weight matrix
weight_matrix = np.zeros((1000, 1000))
rating_matrix = np.zeros((10000, 1000))


def get_indices_of_nearest_neighbors(movie, ratedMovies, k):
    """Get the closest neighbours of a user

    Memoized so that it is not recalculated again and again for the same user
    """
    ind = (-weight_matrix[movie][ratedMovies]).argsort()[:k]
    return ratedMovies[ind]


def predict_rating(user, movie):

    # indices_of_nearest_neigbours = get_indices_of_nearest_neighbors(weight_matrix[user, :], k)
    # nearestNeighbours = np.shape(indices_of_nearest_neigbours)[0]

    numerator, denominator = 0.0, 0.0

    this_user = rating_matrix[user]
    rated_films = np.nonzero(this_user)[0]
    # closest_films = get_indices_of_nearest_neighbors(movie, rated_films, k)
    for movieInd in rated_films:

        if this_user[movieInd] != 0:
            numerator += weight_matrix[movieInd, movie]*this_user[movieInd]

        denominator += weight_matrix[movieInd, movie]

    predicted_rating = numerator / denominator

    return predicted_rating


def nan_dist_items(u, v):
    """Ignore values which are 0 in any of the two vectors"""
    mask = np.logical_and(u!=0, v!=0)
    return cdist([u[mask]], [v[mask]], 'cosine')


def do_prediction(best_k):
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

    weight_matrix = True
    validation_indices_computed = True

    global weight_matrix, averages_of_users, averages_of_movies, valSetSize, numRatings

    trainingSubset = parseInputMatrix()

    """ Partitioning the training data into training and validation"""

    indices_for_validation = generate_validation_set(trainingSubset, indices_for_validation_set_already_chosen=validation_indices_computed)
    print indices_for_validation[:10]
    validationSubset = np.take(trainingSubset, indices_for_validation, axis=0)
    trainingSubset = np.delete(trainingSubset, indices_for_validation, axis=0)


    print("shape of training set: ", trainingSubset.shape)
    print("shape of validation set: ", validationSubset.shape)


    for (u, m, rating) in trainingSubset:
        rating_matrix[u][m] = rating

    print "rating matrix: ", rating_matrix.shape

    if weight_matrix:
        try:
            weight_matrix = np.load('data/knn/weight_matrix_items.npy')
        except Exception as e:
            print "Coundn't load the matrix ", e
            weight_matrix = compute_weight_matrix(rating_matrix=rating_matrix.T, metric=nan_dist_items) 
            np.save('data/knn/weight_matrix_items', weight_matrix)

    else:
        weight_matrix = compute_weight_matrix(rating_matrix=rating_matrix.T, metric=nan_dist_items) 
        np.save('data/knn/weight_matrix_items', weight_matrix)

    k = do_validation(validationSubset)

    do_prediction(k)


if __name__ == '__main__':
    code()