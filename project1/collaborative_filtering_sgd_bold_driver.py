import numpy as np
from general_functions import *

np.random.seed(1729)

n = 10000
m = 1000
c = 40

validationSetPercentage = 0.001

mean_global_rating = 0

#global SGD matrices
P, Q, B_user, B_item = [], [], [], []

def lookupError(r, bias_user, bias_item, p, q):
    return r - predict(bias_user, bias_item, p, q)

def predict(bias_user, bias_item, p, q):
    score = mean_global_rating + bias_user + bias_item + np.dot(p,q)
    # return score if score >=1 and score<=5 else max(1, min(5, score))
    return score

def generateRandomSubsetForCV(cv, dataLen):

    partLen = dataLen/cv

    allIndeces = np.arange(dataLen)
    np.random.shuffle(allIndeces)
    for i in xrange(cv):
        start = i*partLen
        end = (i+1)*partLen
        yield allIndeces[start:end]


def matrix_factorisation(training_data, validation_data, epochs=200, learning_rate=0.001, lam=0.05,p=25.0,m=0.001,
                         inc=0.05,dec=0.75):
    global P, Q, B_item, B_user

    Q = Q.T
    previous_error=100000000.0

    for epoch in xrange(epochs):
        counter = 0
        print("epoch ", epoch)
        print learning_rate
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

        if(e<=previous_error):
            learning_rate *= (1.0+inc)
        else:
            learning_rate *= dec

        previous_error=e

        print np.sqrt(e / np.shape(validation_data)[0])
        print e


def do_validation(validationSubset):
    feature_vector_for_regression = []

    e = 0
    for (i, j, r) in validationSubset:

        prediction = predict(B_user[i], B_item[j], P[i, :], Q[:, j])
        feature_vector_for_regression.append([prediction, r])

        e += pow(lookupError(r, B_user[i], B_item[j], P[i, :], Q[:, j]), 2)

    np.save('data/sgd/feature_vector_sgd', feature_vector_for_regression)
    print np.sqrt(e / np.shape(validationSubset)[0])

def code():
    global P, Q, B_item, B_user
    training_data = np.asarray(parseInputMatrix())

    count = 0
    sum = 0
    for (i, j, k) in training_data:
        count += 1
        sum += k

    mean_global_rating = sum / count

    mask = generate_validation_set(training_data, indices_for_validation_set_already_chosen=True)

    print mask[:10]
    validationSubset = np.take(training_data, mask, axis=0)
    trainingSubset = np.delete(training_data, mask, axis=0)

    print("shape of training set: ", trainingSubset.shape)
    print("shape of validation set: ", validationSubset.shape)


    # initialize matrices
    P = np.random.normal(0, 0.01, (n, c))  #users
    Q = np.random.normal(0, 0.01, (m, c))  #movies

    B_user = np.random.normal(0, 0.01, n)
    B_item = np.random.normal(0, 0.01, m)

    matrix_factorisation(trainingSubset, validationSubset)

    do_validation(validationSubset)

    saveSubmission(P,Q.T, B_user, B_item)



def saveSubmission(nP, nQ, B_user, B_item):
    nR = np.dot(nP, nQ.T)

    np.save('data/sgd/nP_matrix_sgd_200', nP)
    np.save('data/sgd/nQ_matrix_sgd_200', nQ)
    np.save('data/sgd/Bias_user_sgd_200', B_user)
    np.save('data/sgd/Bias_item_sgd_200', B_item)
    np.save('data/sgd/Mean_global_rating_sgd_200', mean_global_rating)

    sample_submission = np.genfromtxt('data/sampleSubmission.csv', delimiter=',', dtype=None)

    f = open('data/sgd/my_prediction_sgd.csv', 'w')

    f.write('%s\n' % "Id,Prediction")

    for i in range(1, np.shape(sample_submission)[0]):
        entry = sample_submission[i][0].split("_")
        row = int(entry[0][1:])
        column = int(entry[1][1:])

        predicted_rating = mean_global_rating + B_user[row - 1] + B_item[column - 1] + nR[row - 1][column - 1]

        if predicted_rating > 5:
            f.write('r%d_c%d,%f\n' % (row, column, 5))
        else:
            f.write('r%d_c%d,%f\n' % (row, column, predicted_rating))

    f.close()


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
    
