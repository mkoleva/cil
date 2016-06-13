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


def get_indices_of_nearest_neighbors(weight_array, k):
    return (-weight_array).argsort()[:k]


def predict_rating(user, movie, k=20):

    indices_of_nearest_neigbours = get_indices_of_nearest_neighbors(weight_matrix[user, :], k)
    nearestNeighbours = np.shape(indices_of_nearest_neigbours)[0]

    numerator, denominator = 0.0, 0.0
    for i in xrange(nearestNeighbours):
        current_neighbour = indices_of_nearest_neigbours[i]

        if rating_matrix[current_neighbour][movie] != 0:
            numerator += weight_matrix[user, current_neighbour] \
                         * (rating_matrix[current_neighbour, movie] - averages_of_users[current_neighbour])

        denominator += weight_matrix[user, current_neighbour]

    """
    rating_nearest_neigbours_normalized = np.subtract(np.take(rating_matrix[:,movie], indices_of_nearest_neigbours), np.take(averages_of_users, indices_of_nearest_neigbours))

    numerator = sum(np.multiply(np.take(weight_matrix[user, :], indices_of_nearest_neigbours), rating_nearest_neigbours_normalized))

    denominator = sum(np.take(weight_matrix[user, :], indices_of_nearest_neigbours))

    """
    predicted_rating = averages_of_users[user] + numerator / denominator

    return predicted_rating

def compute_weight_matrix():
    print "total ratings to iterate: ", np.shape(rating_matrix)[0]
    # pool = multiprocessing.Pool(4)
    # out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))
    from sklearn.metrics.pairwise import pairwise_distances
    weight_matrix = pairwise_distances(rating_matrix, metric='correlation', n_jobs=-1) #user similarity
    weight_matrix = np.nan_to_num(weight_matrix)


    # for i in range(0, np.shape(rating_matrix)[0]):
    #     print i

    #     for j in range(i + 1, np.shape(rating_matrix)[0]):

    #         weight_matrix[i][j] = weight_matrix[j][i] = pearson_distance(i, j)

    np.save('data/knn/weight_matrix', weight_matrix)

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
    best_k = 1
    for k in range(90, 150):
        error = 0
        for (u, m, rating) in validationSubset:
            predicted_rating = predict_rating(u, m, k)
            # print predicted_rating

            error += pow((predicted_rating - rating), 2)

        error /= np.shape(validationSubset)[0]

        print k, np.sqrt(error)

        if error < best_error:
            best_error = error
            best_k = k

    print best_k

    feature_vector_for_regression = []

    for (u, m, rating) in validationSubset:
        predicted_rating = predict_rating(u, m, best_k)
        feature_vector_for_regression.append([predicted_rating, rating])

    np.save('data/knn/feature_vector_knn', feature_vector_for_regression)


    return best_k

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

    averages_of_movies, averages_of_users = computeAverages(rating_matrix)

    if weight_matrix_and_validation_indices_computed == True:
        weight_matrix = np.load('data/knn/weight_matrix.npy')
    else:
        compute_weight_matrix()


    best_k = do_validation(validationSubset)

    do_prediction(best_k)


if __name__ == '__main__':
    code()