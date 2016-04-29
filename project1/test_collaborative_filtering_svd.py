from collaborative_filtering_svd import parseInputMatrix

from pprint import pprint


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
	pprint(bla)
	print expected

if __name__ == '__main__':
    unittest.main()