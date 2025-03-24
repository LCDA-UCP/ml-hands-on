import unittest
import numpy as np
import pandas as pd

from ml_hands_on.data import Dataset
from ml_hands_on.feature_selection import variance_threshold


class  testVariance(unittest.TestCase):

       def test_transform(self):

           X = np.array([['1','4','20','10'],['7','5','20','10'],['0','9','40','10']])
           dataset = Dataset(X,features=)

