import numpy as np
from general_functions import *
from multiprocessing import Process
import time

np.random.seed(1729)

n = 10000
m = 1000
c = 40

mean_global_rating = 0

validationSetPercentage = 0.001
trainingSubset, validationSubset = [], []

def lookupError(r, bias_user, bias_item, p, q):
    return r - predict(bias_user, bias_item, p, q)

def predict(bias_user, bias_item, p, q):
    score =  bias_user + bias_item + np.dot(p,q)
    # return score if score >=1 and score<=5 else max(1, min(5, score))
    return score

def matrix_factorisation(training_data, validation_data, c=80, epochs=200, learning_rate=0.001, lam=0.05,
                         inc=0.05,dec=0.75):

    # initialize matrices
    P = np.random.normal(0, 0.01, (n, c))  #users
    Q = np.random.normal(0, 0.01, (m, c))  #movies

    B_user = np.random.normal(0, 0.01, n)
    B_item = np.random.normal(0, 0.01, m)

    Q = Q.T
    previous_error=100000000.0

    for epoch in xrange(epochs):
        begin = time.time()
        counter = 0

        np.random.shuffle(training_data)

        for (i, j, r) in training_data:
            # if counter % 10000 == 0:
                # print counter
            counter += 1

            eij = lookupError(r, B_user[i], B_item[j], P[i, :], Q[:, j])

            # for the full feature vector
            nP = P[i, :]

            temp_update_P= learning_rate*(eij*Q[:, j] - lam*P[i,:])
            P[i, :] += temp_update_P

            temp_update_Q=learning_rate*(eij*nP - lam*Q[:,j])
            Q[:,j] += temp_update_Q

            temp_B_user_update=learning_rate*(eij - lam*B_user[i])
            B_user[i] += temp_B_user_update

            temp_B_item_update=learning_rate*(eij - lam*B_item[j])
            B_item[j] += temp_B_item_update


        e = 0
        testCounter = 0

        for (i, j, r) in validation_data:
            e += pow(lookupError(r, B_user[i], B_item[j], P[i, :], Q[:, j]), 2)

        #e = np.sqrt(e / np.shape(validation_data)[0])
        end = time.time()

        print("SGD epoch %d, error = %f,"
              " time = %.2fs learning rate = %f c = %f"
              % ((epoch+1),
                 np.sqrt(e / np.shape(validation_data)[0]), end - begin, learning_rate, c))

        if(e<=previous_error):
            learning_rate *= (1.0+inc)
        else:
            learning_rate *= dec

        previous_error=e

    return P, Q, B_user, B_item


def do_validation(validationSubset, P, Q, B_user, B_item, c_used_for_printing):
    feature_vector_for_regression = []

    e = 0
    for (i, j, r) in validationSubset:

        prediction = predict(B_user[i], B_item[j], P[i, :], Q[:, j])
        feature_vector_for_regression.append([prediction, r])

        e += pow(lookupError(r, B_user[i], B_item[j], P[i, :], Q[:, j]), 2)

    np.save('data/sgd/feature_vector_sgd_'+str(c_used_for_printing), feature_vector_for_regression)
    print np.sqrt(e / np.shape(validationSubset)[0])

def do_prediction(nP, nQ, B_user, B_item, c_used_for_printing):
    nR = np.dot(nP, nQ.T)

    sample_submission = np.genfromtxt('data/sampleSubmission.csv', delimiter=',', dtype=None)

    f = open('data/sgd/my_prediction_sgd_'+str(c_used_for_printing)+'.csv', 'w')

    f.write('%s\n' % "Id,Prediction")

    for i in range(1, np.shape(sample_submission)[0]):
        entry = sample_submission[i][0].split("_")
        row = int(entry[0][1:])
        column = int(entry[1][1:])

        predicted_rating =  B_user[row - 1] + B_item[column - 1] + nR[row - 1][column - 1]

        if predicted_rating > 5:
            f.write('r%d_c%d,%f\n' % (row, column, 5))
        else:
            f.write('r%d_c%d,%f\n' % (row, column, predicted_rating))

    f.close()

def task1():
    global trainingSubset, validationSubset

    P, Q, B_user, B_item = matrix_factorisation(trainingSubset, validationSubset, c=80, epochs=98,inc=0.05,dec=0.7)
    do_validation(validationSubset, P, Q, B_user, B_item,80)
    do_prediction(P,Q.T, B_user, B_item,80)

def task2():
    global trainingSubset, validationSubset

    P, Q, B_user, B_item = matrix_factorisation(trainingSubset, validationSubset, c=100, epochs=105,inc=0.05,dec=0.7)
    do_validation(validationSubset, P, Q, B_user, B_item,100)
    do_prediction(P,Q.T, B_user, B_item,100)


def run_2_sgds_in_parallel():

    p1 = Process(target=task1)
    p2 = Process(target=task2)

    p1.start()
    p2.start()

def code():
    global trainingSubset, validationSubset

    training_data = np.asarray(parseInputMatrix())

    count = 0
    sum = 0
    for (i, j, k) in training_data:
        count += 1
        sum += k

    mask = generate_validation_set(training_data, indices_for_validation_set_already_chosen=True)

    print mask[:10]
    validationSubset = np.take(training_data, mask, axis=0)
    trainingSubset = np.delete(training_data, mask, axis=0)

    print("shape of training set: ", trainingSubset.shape)
    print("shape of validation set: ", validationSubset.shape)

    run_2_sgds_in_parallel()


def saveSubmission(nP, nQ, B_user, B_item):
    np.save('data/sgd/nP_matrix_sgd_200', nP)
    np.save('data/sgd/nQ_matrix_sgd_200', nQ)
    np.save('data/sgd/Bias_user_sgd_200', B_user)
    np.save('data/sgd/Bias_item_sgd_200', B_item)
    np.save('data/sgd/Mean_global_rating_sgd_200', mean_global_rating)

    do_prediction(nP, nQ, B_user, B_item)

def generateRandomSubsetForCV(cv, dataLen):

    partLen = dataLen/cv

    allIndeces = np.arange(dataLen)
    np.random.shuffle(allIndeces)
    for i in xrange(cv):
        start = i*partLen
        end = (i+1)*partLen
        yield allIndeces[start:end]

def crossValidationRun(cv=10):
    training_data = np.asarray(parseInputMatrix())

    numRatings = len(training_data)
    validationSetPercentage = int(1/cv)

    cvCounter = 0
    errors = []

    for cvCounter, mask in enumerate(generateRandomSubsetForCV(cv=cv, dataLen=numRatings)):
        print "/////////////"
        print "CV ", cvCounter
        validationSubset = np.take(training_data, mask, axis=0)
        trainingSubset = np.delete(training_data, mask, axis=0)

        print("shape of training set: ", trainingSubset.shape)
        print("shape of validation set: ", validationSubset.shape)

        # initialize matrices
        P = np.random.normal(0, 0.01, (n, c))  #users
        Q = np.random.normal(0, 0.01, (m, c))  #movies

        P, Q, error = matrix_factorisation(P, Q, trainingSubset, validationSubset)
        errors.append(error)

    print "FINAL RESULT"
    print errors
    print sum(errors)/cvCounter


if __name__ == '__main__':
    code()
    # crossValidationRun()

# 1.01125 with 50 epochs, initial learning rate 0.002, p=25
# 1.01213098695 with 50 epochs, initial learning rate 0.0015, p=25, m=0.001
# 1.01212951839 with 50 epochs, initial learning rate 0.001, bold driver, 0.05, 0.5
# 1.01037583231 with 80 epochs, initial learning rate 0.001, bold driver, 0.05, 0.75
    
