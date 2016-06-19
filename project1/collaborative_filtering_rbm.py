__author__ = 'karimlabib'

import numpy as np

from general_functions import *
from sklearn.neural_network import BernoulliRBM
from sklearn.grid_search import GridSearchCV
from rbm_custom import BernoulliRBM_Custom

machine = BernoulliRBM_Custom
trainingMatrix = []
visible_input_probability = []

def compute_visible_input_probability():
    global visible_input_probability, trainingMatrix, machine

    visible_input_probability = np.zeros((10000, 5000))

    p_hat =  machine.transform(trainingMatrix)
    visible_input_probability = machine._mean_visibles(p_hat)

def predict_rating(user, movie):
    global machine, visible_input_probability

    tmp = visible_input_probability[user]

    numerator = np.exp(tmp[movie*5: (movie+1)*5])
    denominator = sum(numerator)

    predicted_rating = numerator / denominator

    return_value = 0
    for i in range(1,6):
        return_value += i*predicted_rating[i-1]

    return return_value

def do_validation(validationSubset, save_feature_vector=False):

    compute_visible_input_probability()

    feature_vector_for_regression = []

    error = 0

    for (u, m, rating) in validationSubset:

        predicted_rating = predict_rating(u, m)

        feature_vector_for_regression.append([predicted_rating, rating])

        error += pow((predicted_rating - rating), 2)


    error /= np.shape(validationSubset)[0]

    error = np.sqrt(error)

    if save_feature_vector:
        np.save('data/rbm/feature_vector_rbm', feature_vector_for_regression)

    return error


def do_prediction():
    sample_submission = np.genfromtxt('data/sampleSubmission.csv', delimiter=',', dtype=None)

    f = open('data/rbm/my_prediction_rbm.csv', 'w')

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

def get_index_of_movie_rating_in_visible_vector(movie, rating):
    return 5 * movie + (rating - 1)

def code():

    global trainingMatrix, machine

    trainingSubset = parseInputMatrix()

    """ Partitioning the training data into training and validation"""

    indices_for_validation = generate_validation_set(trainingSubset, indices_for_validation_set_already_chosen=True)
    print indices_for_validation[:10]
    validationSubset = np.take(trainingSubset, indices_for_validation, axis=0)
    trainingSubset = np.delete(trainingSubset, indices_for_validation, axis=0)


    print("shape of training set: ", trainingSubset.shape)
    print("shape of validation set: ", validationSubset.shape)

    trainingMatrix = np.zeros((10000, 5000))

    for (u, m, rating) in trainingSubset:
        trainingMatrix[u][get_index_of_movie_rating_in_visible_vector(m, rating)] = 1

    # calculated by grid search below
    min_error = 0.998
    best_hidden = 80
    best_iter = 19

    if False:
        for num_hidden in range(20, 101, 10):
            for num_iter in range(10, 20):
                machine = BernoulliRBM(random_state=0, verbose=True, n_components=num_hidden,
                                       n_iter=num_iter, learning_rate=0.05)
                machine.fit(trainingMatrix)
                print num_hidden, num_iter
                error = do_validation(validationSubset)

                if error < min_error:
                    min_error = error
                    best_hidden = num_hidden
                    best_iter = num_iter

                print "best: ", min_error, best_hidden, best_iter

    machine = BernoulliRBM_Custom(random_state=0, verbose=True, n_components=best_hidden,
                                 n_iter=100, learning_rate=0.2, error_function=do_validation,
                                 validation_set=validationSubset)
    machine.fit(trainingMatrix)
    print best_hidden, best_iter

    do_validation(validationSubset, save_feature_vector= True)

    do_prediction()


if __name__ == '__main__':
    code()
