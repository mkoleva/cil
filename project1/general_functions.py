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


def plotRBM():
    import matplotlib.pyplot as plt

    dataB = np.load('data/rbm/rbm_c_100_with_bold_driver.npy')[:,1]
    data = np.load('data/rbm/rbm_c_100_without_bold_driver.npy')[:,1]
    dataB2 = np.load('data/rbm/rbm_c_80_with_bold_driver.npy')[:,1]
    data2 = np.load('data/rbm/rbm_c_80_without_bold_driver.npy')[:,1]


    # with plt.style.context('fivethirtyeight'):

    minBold100 = np.argmin(dataB)
    minBold80 = np.argmin(dataB2)
    min80 = np.argmin(data2)
    min100 = np.argmin(data)

    minBold100Val = min(dataB)
    minBold80Val = min(dataB2)
    min80Val = min(data2)
    min100Val = min(data)


    print np.argmin(dataB), min(dataB)
    print np.argmin(dataB2), min(dataB2)

    print np.argmin(data), min(data)
    print np.argmin(data2), min(data2)

    plt.xlabel('Epochs')
    plt.ylabel('RMSE')

    plt.plot(dataB, linestyle='-.', linewidth=2, label='BD 100 hidden')
    plt.plot(dataB2, linestyle='--', linewidth=2, label='BD 80 hidden')
    plt.plot(data, linestyle='-', linewidth=1, label='100 hidden')
    plt.plot(data2, linestyle='-', linewidth=1, label='80 hidden')


    plt.text(minBold80+1, 1.038, 'best BD80')
    plt.text(minBold100+1, 1.035, 'best BD100')
    plt.text(min80+1, 1.035, 'best c80')
    plt.text(min100+1, 1.030, 'best c100')


    plt.axvline(x=minBold80, color='gray', alpha=0.7)
    plt.axvline(x=minBold100, color='gray')
    plt.axvline(x=min80, color='gray', alpha=0.7)
    plt.axvline(x=min100, color='gray', alpha=0.7)

    # plt.axhline(y=minBold80Val, color='gray', alpha=0.7)
    # plt.axhline(y=minBold100Val, color='gray')
    # plt.axhline(y=min80Val, color='gray', alpha=0.7)
    # plt.axhline(y=min100Val, color='gray', alpha=0.7)


    plt.legend(loc='upper left')

    plt.show()

    # for point1, point2 in zip(data, data2):
    #     print str(point1) + '\t' + str(point2)


def plotSGD():
    import matplotlib.pyplot as plt

    dataB = np.sqrt(np.load('data/sgd/error_vector_sgd_bold_driver.npy')/58847)
    dataB2 = np.sqrt(np.load('data/sgd/error_vector_sgd_bold_driver_80.npy')/58847)
    dataB3 = np.sqrt(np.load('data/sgd/error_vector_sgd_bold_driver_100.npy')/58847)
    data2 = np.sqrt(np.load('data/sgd/error_vector_sgd_without_80.npy')/58847)
    data = np.sqrt(np.load('data/sgd/error_vector_sgd_without.npy')/58847)
    data3 = np.sqrt(np.load('data/sgd/error_vector_sgd_without_100.npy')/58847)


    # with plt.style.context('fivethirtyeight'):

    minBold40 = np.argmin(dataB)
    minBold80 = np.argmin(dataB2)
    minBold100 = np.argmin(dataB3)
    min80 = np.argmin(data2)
    min40 = np.argmin(data)
    min100 = np.argmin(data3)


    print np.argmin(dataB), min(dataB)
    print np.argmin(dataB2), min(dataB2)
    print np.argmin(dataB3), min(dataB3)

    print np.argmin(data), min(data)
    print np.argmin(data2), min(data2)
    print np.argmin(data3), min(data3)

    plt.xlabel('Epochs')
    plt.ylabel('RMSE')

    plt.plot(dataB, linestyle='-.', linewidth=2, label='BD c=40')
    plt.plot(dataB2, linestyle='--', linewidth=2, label='BD c=80')
    plt.plot(dataB3, linestyle=':', linewidth=2, label='BD c=100')
    plt.plot(data, linestyle='-', linewidth=1, label='c=40')
    plt.plot(data2, linestyle='-', linewidth=1, label='c=80')
    plt.plot(data3, linestyle='-', linewidth=1, label='c=100')


    plt.text(minBold80+1, 1.4, 'best BD80')
    plt.text(minBold40+1, 1.4, 'best BD40')
    plt.text(minBold100+1, 1.35, 'best BD100')
    plt.text(min80+1, 1.35, 'best c80')
    plt.text(min40+1, 1.31, 'best c40')
    plt.text(min100+1, 1.33, 'best c100')


    plt.axvline(x=minBold80, color='gray', alpha=0.7)
    plt.axvline(x=minBold40, color='gray')
    plt.axvline(x=minBold100, color='gray')
    plt.axvline(x=min40, color='gray', alpha=0.7)
    plt.axvline(x=min80, color='gray', alpha=0.7)
    plt.axvline(x=min100, color='gray', alpha=0.7)

    plt.legend(loc='upper left')

    plt.show()

    # for point1, point2 in zip(data, data2):
    #     print str(point1) + '\t' + str(point2)



if __name__=='__main__':
    # plotStuff()
    plotRBM()




