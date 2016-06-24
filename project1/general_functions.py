__author__ = 'karimlabib'

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def generateRandomSubset(len, upTo):
    allIndeces = np.arange(upTo)
    np.random.shuffle(allIndeces)
    return allIndeces[:len]

def generate_validation_set(trainingSubset, indices_for_validation_set_already_chosen=False):

    validationSetPrecentage = 0.05

    numRatings = len(trainingSubset)
    valSetSize = int(numRatings*validationSetPrecentage)

    print("Validation set size:", valSetSize)

    indices_for_validation = []
    if indices_for_validation_set_already_chosen == True:
        indices_for_validation = np.load('data/general/indices_for_validation_set.npy')
    else:
        indices_for_validation = generateRandomSubset(len=valSetSize, upTo=numRatings)
        np.save('data/general/indices_for_validation_set', indices_for_validation)

    return indices_for_validation

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
    """ Calculates mean rating for a film across users and mean rating of user across films."""

    numUsers, numFilms = matrix.shape

    # Finds the mean across every column, ignoring 0s and then sets nan to 0
    averages_of_movies = np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]), 0, matrix)
    averages_of_movies[np.isnan(averages_of_movies)] = 0.

    # Finds the mean across every row, ignoring 0s and then sets nan to 0
    averages_of_users = np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]), 1, matrix)
    averages_of_users[np.isnan(averages_of_users)] = 0.

    return averages_of_movies, averages_of_users


def compute_weight_matrix(rating_matrix, metric):
    """ Computes a pairwise similarity matrix for the samples in the matrix based on the metric f-n provided"""
    
    dist_matrix = pairwise_distances(rating_matrix, metric=metric, n_jobs=-1) #user similarity
    dist_matrix = np.nan_to_num(dist_matrix)
    print "Some distances ", dist_matrix[0]

    weight_matrix = 1.0-dist_matrix  # convert to weight matrix from dist matrix
    return weight_matrix

def sort_indices_of_neighbors(weight_matrix):
    sorted_indices_of_neighbors = (-weight_matrix).argsort()
    return sorted_indices_of_neighbors
