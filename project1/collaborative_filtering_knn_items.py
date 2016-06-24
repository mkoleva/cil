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

averages_of_users, averages_of_movies = [], []
sorted_indices_of_neighbors = []

def memodict(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__

@memodict
def get_indices_of_nearest_neighbors(params):
    """Get the closest neighbours of a movie

    Memoized so that it is not recalculated again and again for the same movie
    """
    movie, k = params

    return (-weight_matrix[movie,:]).argsort()[:k]

def predict_rating(user, movie, k):
    global rating_matrix, weight_matrix, averages_of_users, averages_of_movies, sorted_indices_of_neighbors

    numerator, denominator = 0.0, 0.0

    this_user = rating_matrix[user]
    rated_films = np.nonzero(this_user)[0]
    closest_films = sorted_indices_of_neighbors[movie]

    counter_of_neighbors = 0

    for movieInd in closest_films:

        if counter_of_neighbors == k:
            break

        if this_user[movieInd] != 0:
            counter_of_neighbors += 1

            numerator += weight_matrix[movieInd, movie] * (this_user[movieInd] - averages_of_movies[movieInd])

            denominator += weight_matrix[movieInd, movie]

    predicted_rating = averages_of_movies[movie]
    if denominator != 0:
        predicted_rating = averages_of_movies[movie] + numerator / denominator



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

        predicted_rating = predict_rating(user - 1, movie - 1, best_k)

        if predicted_rating > 5:
            f.write('r%d_c%d,%f\n' % (user, movie, 5))
        else:
            f.write('r%d_c%d,%f\n' % (user, movie, predicted_rating))

    f.close()

def do_validation(validationSubset):
    best_error = 10
    best_k = 20

    if True :
        save_for_validation = []
        save_k = []
        for k in range(10, 100, 10):
            error = 0
            for (u, m, rating) in validationSubset:
                predicted_rating = predict_rating(u, m, k)

                error += pow((predicted_rating - rating), 2)

            error /= np.shape(validationSubset)[0]
            error = np.sqrt(error)

            save_k.append(k)
            save_for_validation.append(error)

            print k, error

            if error < best_error:
                best_error = error
                best_k = k

        np.save('data/knn/save_knn_item_k_validation_for_plot', save_k)
        np.save('data/knn/save_knn_item_error_validation_for_plot', save_for_validation)

    print best_k

    error = 0

    feature_vector_for_regression = []

    for (u, m, rating) in validationSubset:
        predicted_rating = predict_rating(u, m, best_k)
        error += pow((predicted_rating - rating), 2)
        feature_vector_for_regression.append([predicted_rating, rating])

    error /= np.shape(validationSubset)[0]
    print np.sqrt(error)

    np.save('data/knn/feature_vector_knn_items', feature_vector_for_regression)


def code():

    weight_matrix_computed = True
    validation_indices_computed = True

    global weight_matrix, averages_of_users, averages_of_movies, sorted_indices_of_neighbors

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

    averages_of_movies, averages_of_users = computeAverages(rating_matrix)

    # missing values will go from 0 to negative, convert back to 0
    mask_of_unrated_items = np.where(rating_matrix==0)
    tmp_matrix_of_averages = np.zeros((10000,1000))
    tmp_matrix_of_averages = tmp_matrix_of_averages + averages_of_movies[:, np.newaxis].T

    # remove means for every user at every location where he rated a movie.. unrated movies should stay 0
    tmp_matrix_of_averages[mask_of_unrated_items] = 0

    centered_rating_matrix = rating_matrix - tmp_matrix_of_averages

    if weight_matrix_computed:
        try:
            weight_matrix = np.load('data/knn/weight_matrix_items.npy')
        except Exception as e:
            print "Could not load the matrix ", e
            weight_matrix = compute_weight_matrix(rating_matrix=rating_matrix.T, metric=nan_dist_items) 
            np.save('data/knn/weight_matrix_items', weight_matrix)

    else:
        weight_matrix = compute_weight_matrix(rating_matrix=rating_matrix.T, metric=nan_dist_items) 
        np.save('data/knn/weight_matrix_items', weight_matrix)

    sorted_indices_of_neighbors = sort_indices_of_neighbors(weight_matrix)

    k = do_validation(validationSubset)

    do_prediction(k)


if __name__ == '__main__':
    code()