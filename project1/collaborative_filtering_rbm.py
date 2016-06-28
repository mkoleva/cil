__author__ = 'karimlabib'

import numpy as np

from general_functions import *
from sklearn.neural_network import BernoulliRBM
from sklearn.grid_search import GridSearchCV
from rbm_custom import BernoulliRBM_Custom

from multiprocessing import Process

machine = BernoulliRBM_Custom
trainingMatrix = []

machine_80_bd, machine_100_bd = BernoulliRBM_Custom, BernoulliRBM_Custom
machine_80_no_bd, machine_100_no_bd = BernoulliRBM_Custom, BernoulliRBM_Custom

def do_prediction(machine):

    visible_input_probability = machine.compute_visible_input_probability(trainingMatrix)

    sample_submission = np.genfromtxt('data/sampleSubmission.csv', delimiter=',', dtype=None)

    f = open('data/rbm/my_prediction_rbm.csv', 'w')

    f.write('%s\n' % "Id,Prediction")

    for i in range(1, np.shape(sample_submission)[0]):
        if i % 1000 == 0:
            print i

        entry = sample_submission[i][0].split("_")
        user = int(entry[0][1:])
        movie = int(entry[1][1:])

        predicted_rating = machine.predict_rating(user - 1, movie - 1, visible_input_probability)

        if predicted_rating > 5:
            f.write('r%d_c%d,%f\n' % (user, movie, 5))
        else:
            f.write('r%d_c%d,%f\n' % (user, movie, predicted_rating))

    f.close()

def get_index_of_movie_rating_in_visible_vector(movie, rating):
    return 5 * movie + (rating - 1)

def task_1():
    machine_80_bd.fit(trainingMatrix)

def task_2():
    machine_100_bd.fit(trainingMatrix)

def task_3():
    machine_80_no_bd.fit(trainingMatrix)

def task_4():
    machine_100_no_bd.fit(trainingMatrix)

def do_some_plots(validationSubset):
    global machine_80_bd, machine_100_bd, machine_100_no_bd, machine_80_no_bd

    machine_80_bd = BernoulliRBM_Custom(random_state=0, verbose=True, n_components=80,
                                     n_iter=60, learning_rate=0.05, validation_set=validationSubset,
                                     use_bold_driver=True)

    machine_100_bd = BernoulliRBM_Custom(random_state=0, verbose=True, n_components=100,
                                      n_iter=60, learning_rate=0.05, validation_set=validationSubset,
                                      use_bold_driver=True)

    machine_80_no_bd = BernoulliRBM_Custom(random_state=0, verbose=True, n_components=80,
                                     n_iter=60, learning_rate=0.05, validation_set=validationSubset,
                                     use_bold_driver=False)

    machine_100_no_bd = BernoulliRBM_Custom(random_state=0, verbose=True, n_components=100,
                                      n_iter=60, learning_rate=0.05, validation_set=validationSubset,
                                      use_bold_driver=False)

    p1 = Process(target=task_1)
    p2 = Process(target=task_2)
    p3 = Process(target=task_3)
    p4 = Process(target=task_4)

    p1.start()
    p2.start()
    p3.start()
    p4.start()

def do_grid_search():
    for num_hidden in range(20, 101, 10):
            for num_iter in range(10, 20):
                machine = BernoulliRBM(random_state=0, verbose=True, n_components=num_hidden,
                                       n_iter=num_iter, learning_rate=0.05, validation_set=validationSubset,
                                       use_bold_driver=True)
                machine.fit(trainingMatrix)
                print num_hidden, num_iter
                error = do_validation(validationSubset)

                if error < min_error:
                    min_error = error
                    best_hidden = num_hidden
                    best_iter = num_iter

                print "best: ", min_error, best_hidden, best_iter

def code():

    global trainingMatrix

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

    #do_some_plots(validationSubset)

    machine = BernoulliRBM_Custom(random_state=0, verbose=True, n_components=50,
                                  n_iter=60, learning_rate=0.05, validation_set=validationSubset,
                                  use_bold_driver=True)

    machine.fit(trainingMatrix)

    machine.do_validation(trainingMatrix, validationSubset, save_feature_vector= True)

    do_prediction(machine)


if __name__ == '__main__':
    code()
