__author__ = 'karimlabib'

import numpy as np

from general_functions import *
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import correlation as corr_dist
from scipy.spatial.distance import pdist, cdist


amount_of_validation = 0.01
"""this is the pearson distance global weight matrix"""
weight_matrix = np.zeros((10000, 10000))
averages_of_movies, averages_of_users = [], []
rating_matrix = np.zeros((10000, 1000))
centered_rating_matrix = np.zeros((10000, 1000))


movies_indices_rated_by_user = [set() for _ in xrange(10000)]

def memodict(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret 
    return memodict().__getitem__


def get_intersection(u1, u2):
    """take two vectors of ratings with length equal to total number of movies and return a list of
    indices of co-rated movies"""

    return list(movies_indices_rated_by_user[u1] & movies_indices_rated_by_user[u2])


def pearson_distance(u1, u2):
    intersection_list = get_intersection(u1, u2)
    interesectedFilms = np.shape(intersection_list)[0]

    if interesectedFilms == 0:
        return 0

    average_of_user_1, average_of_user_2 = 0, 0

    u1_ratings_in_intersection = np.take(rating_matrix[u1], intersection_list)
    u2_ratings_in_intersection = np.take(rating_matrix[u2], intersection_list)

    average_of_user_1 = np.mean(u1_ratings_in_intersection)
    average_of_user_2 = np.mean(u2_ratings_in_intersection)

    normalized_user_1 = np.subtract(u1_ratings_in_intersection, average_of_user_1)
    normalized_user_2 = np.subtract(u2_ratings_in_intersection, average_of_user_2)

    numerator = sum(np.multiply(normalized_user_1, normalized_user_2))

    denominator1 = sum(np.multiply(normalized_user_1, normalized_user_1))

    denominator2 = sum(np.multiply(normalized_user_2, normalized_user_2))

    denominator = np.sqrt(denominator1 * denominator2)

    if denominator == 0:
        return 0

    return numerator / denominator

@memodict
def get_indices_of_nearest_neighbors(params):
    """Get the closest neighbours of a user

    Memoized so that it is not recalculated again and again for the same user
    """

    user, k = params
    return (-weight_matrix[user, :]).argsort()[:k]


def predict_rating(user, movie, k=20):
    global rating_matrix, weight_matrix, averages_of_users

    indices_of_nearest_neigbours = get_indices_of_nearest_neighbors((user, k))

    numerator, denominator = 0.0, 0.0
    for current_neighbour in indices_of_nearest_neigbours:
        if rating_matrix[current_neighbour, movie] != 0:
            numerator += weight_matrix[user, current_neighbour] \
                         * (rating_matrix[current_neighbour, movie] - averages_of_users[current_neighbour])

            denominator += abs(weight_matrix[user, current_neighbour])

    predicted_rating = averages_of_users[user]
    if denominator != 0:
        predicted_rating = averages_of_users[user] + numerator / denominator

    return predicted_rating

def nan_dist_users(u, v):
    """ Ignore missing values and then calculate dist"""

    mask = np.logical_and(u!=0, v!=0)
    return cdist([u[mask]], [v[mask]], 'cosine')


def do_prediction(best_k):
    sample_submission = np.genfromtxt('data/sampleSubmission.csv', delimiter=',', dtype=None)

    f = open('data/knn/my_prediction_knn.csv', 'w')

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
    best_k = 3500

    if False:
        save_for_validation = []
        save_k = []
        for k in range(1000, 10000, 500):
            error = 0
            for (u, m, rating) in validationSubset:
                predicted_rating = predict_rating(u, m, k)
                # print predicted_rating

                error += pow((predicted_rating - rating), 2)

            error /= np.shape(validationSubset)[0]
            error = np.sqrt(error)

            save_k.append(k)
            save_for_validation.append(error)

            print k, error

            if error < best_error:
                best_error = error
                best_k = k

        np.save('data/knn/save_knn_user_k_validation_for_plot', save_k)
        np.save('data/knn/save_knn_user_error_validation_for_plot', save_for_validation)

    print best_k

    feature_vector_for_regression = []

    for (u, m, rating) in validationSubset:
        predicted_rating = predict_rating(u, m, best_k)
        feature_vector_for_regression.append([predicted_rating, rating])

    np.save('data/knn/feature_vector_knn', feature_vector_for_regression)


    return best_k

def code():

    weight_matrix_computed = True
    indices_for_validation_computed = True

    global rating_matrix, weight_matrix, averages_of_users, averages_of_movies

    trainingSubset = parseInputMatrix()

    """ Partitioning the training data into training and validation"""

    indices_for_validation = generate_validation_set(trainingSubset, indices_for_validation_set_already_chosen=indices_for_validation_computed)
    print indices_for_validation[:10]
    validationSubset = np.take(trainingSubset, indices_for_validation, axis=0)
    trainingSubset = np.delete(trainingSubset, indices_for_validation, axis=0)


    print("shape of training set: ", trainingSubset.shape)
    print("shape of validation set: ", validationSubset.shape)


    for (u, m, rating) in trainingSubset:
        movies_indices_rated_by_user[u].add(m)
        rating_matrix[u][m] = rating

    averages_of_movies, averages_of_users = computeAverages(rating_matrix)

    # missing values will go from 0 to negative, convert back to 0
    mask_of_unrated_items = np.where(rating_matrix==0)
    tmp_matrix_of_averages = np.zeros((10000,1000))
    tmp_matrix_of_averages = tmp_matrix_of_averages + averages_of_users[:, np.newaxis]

    # remove means for every user at every location where he rated a movie.. unrated movies should stay 0
    tmp_matrix_of_averages[mask_of_unrated_items] = 0

    centered_rating_matrix = rating_matrix - tmp_matrix_of_averages

    if weight_matrix_computed:
        try:
            weight_matrix = np.load('data/knn/weight_matrix.npy')
        except Exception as e:
            print "Coundn't load the matrix ", e
            weight_matrix = compute_weight_matrix(rating_matrix=centered_rating_matrix, metric=nan_dist_users)
            np.save('data/knn/weight_matrix', weight_matrix)

    else:
        weight_matrix = compute_weight_matrix(rating_matrix=centered_rating_matrix, metric=nan_dist_users)
        np.save('data/knn/weight_matrix', weight_matrix)

    best_k = do_validation(validationSubset)

    do_prediction(best_k)


if __name__ == '__main__':
    code()