from collaborative_filtering_svd import parseInputMatrix, computeAverages
from pprint import pprint
import numpy as np
import unittest

class TestMatrixStuff(unittest.TestCase):

    def test_parseInputMatrix(self):
        """ converts rRowId_cColID, score in a list of (rowid, colid, score) tuples"""
        bla = parseInputMatrix(path='data/test_data_train.csv')
        expected = [
        (45, 0, 4),
        (58, 0, 3),
        (192, 0, 4),
        (4999, 1, 5),
        (6000, 4, 5)
        ]
        self.assertListEqual(bla, expected)
        # pprint(bla)
        # pprint(expected)

    def test_computeAverages(self):
        test_data = np.array([[10, 0, 10, 0], [1, 1, 0, 0], [9, 9, 9, 0], [0, 10, 1, 0]])
#        [[10,  0, 10,  0],
#        [ 1,  1,  0,  0],
#        [ 9,  9,  9,  0],
#        [ 0, 10,  1,  0]])
        average_mov, average_users = computeAverages(test_data)
        print average_mov, average_users



if __name__ == '__main__':
    unittest.main()