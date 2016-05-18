import numpy as np
from collaborative_filtering_svd import parseInputMatrix



n = 10000
m = 1000
c = 40

validationSetPercentage = 0.001


def memodict(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret 
    return memodict().__getitem__


# @memodict
def lookupError(r,p,q):
    return r - predict(p,q)

def predict(p,q):
    score = np.dot(p,q)
    # return score if score >=1 and score<=5 else max(1, min(5, score))
    return score

def generateRandomSubset(len, upTo):
    allIndeces = np.arange(upTo)
    np.random.shuffle(allIndeces)
    return allIndeces[:len]


def generateRandomSubsetForCV(cv, dataLen):

    partLen = dataLen/cv

    allIndeces = np.arange(dataLen)
    np.random.shuffle(allIndeces)
    for i in xrange(cv):
        start = i*partLen
        end = (i+1)*partLen
        yield allIndeces[start:end]


def matrix_factorisation(P, Q,  training_data, 
    validation_data, epochs=70, learning_rate=0.001, 
    lam=0.05):

    Q = Q.T

    for epoch in xrange(epochs):
        counter = 0
        print("epoch ", epoch)

        # training_data = zip(user_training, movie_training, rating_training)
        np.random.shuffle(training_data)

        for (i, j, r) in training_data:
            # if counter % 10000 == 0:
                # print counter
            counter += 1

            eij = lookupError(r,P[i, :], Q[:, j])


            # for the full feature vector
            nP = P[i, :]
            P[i, :] += learning_rate*(eij*Q[:,j] - lam*P[i,:])
            Q[:,j] += learning_rate*(eij*nP - lam*Q[:,j])

            P[i, :] = nP

        e = 0
        testCounter = 0

        for (i, j, r) in validation_data:
            testCounter += 1
            # print r, np.dot(P[i,:], Q[:,j])
            e += pow(lookupError(r, P[i, :], Q[:, j]), 2)

        print np.sqrt(e / testCounter)
        print e, testCounter


    return P, Q.T, np.sqrt(e / testCounter)
    



def code():
    training_data = np.asarray(parseInputMatrix())
    numRatings = len(training_data)
    valSetSize = int(numRatings*validationSetPercentage)
    print("Validation set size:", valSetSize)
    print training_data
    mask = generateRandomSubset(len=valSetSize, upTo=numRatings)

    print mask[:10]
    validationSubset = np.take(training_data, mask, axis=0)
    trainingSubset = np.delete(training_data, mask, axis=0)

    print("shape of training set: ", trainingSubset.shape)
    print("shape of validation set: ", validationSubset.shape)


    # initialize matrices
    P = np.random.normal(0, 0.01, (n, c))  #users
    Q = np.random.normal(0, 0.01, (m, c))  #movies

    P, Q, e = matrix_factorisation(P, Q, trainingSubset, validationSubset)

    saveSubmission(P,Q)



def saveSubmission(nP, nQ):
    nR = np.dot(nP, nQ.T)

    np.save('nP_matrix', nP)
    np.save('nQ_matrix', nQ)

    sample_submission = np.genfromtxt('data/sampleSubmission.csv', delimiter=',', dtype=None)

    f = open('data.my_predictionM.csv', 'w')

    f.write('%s\n' % "Id,Prediction")

    for i in range(1, np.shape(sample_submission)[0]):
        entry = sample_submission[i][0].split("_")
        row = int(entry[0][1:])
        column = int(entry[1][1:])

        if nR[row - 1][column - 1] > 5:
            f.write('r%d_c%d,%f\n' % (row, column, 5))
        else:
            f.write('r%d_c%d,%f\n' % (row, column, nR[row - 1][column - 1]))

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


    