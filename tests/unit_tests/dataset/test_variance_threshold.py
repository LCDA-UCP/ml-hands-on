import unittest
import numpy as np
import pandas as pd

from ml_hands_on import VarianceThreshold
from ml_hands_on.data import Dataset


class  test_varience_threshold(unittest.TestCase):

        def test_fit(self):

           X = np.array([[1,2,3],[1,2,3],[1,2,3]])

           dataset = Dataset(X=X, y =None,features=['f1','f2','f3'])

           vt = VarianceThreshold()
           vt._fit(dataset)

           expected_variance = np.var(X, axis=0)
           assert np.array_equal(vt.variance, expected_variance)



        def test_transform(self):
            X = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 5]])
            dataset = Dataset(X=X, y=None, features=["f1", "f2", "f3"])

            vt = VarianceThreshold(threshold=0.2)
            vt._fit(dataset)
            new_dataset = vt._transform(dataset)

            assert new_dataset.X.shape[1] == 1
            assert new_dataset.features == ["f3"]


        def test_transform_no_removal(self):
            X = np.array([[6, 2, 4], [1, 26, 15], [3, 9, 5]])
            dataset = Dataset(X=X, y=None, features=["f1", "f2", "f3"])

            vt = VarianceThreshold(threshold=0.5)
            vt._fit(dataset)
            new_dataset = vt._transform(dataset)

            assert new_dataset.X.shape[1]== 3
            assert new_dataset.features == ["f1", "f2", "f3"]

        def test_transform_all_removed(self):
            X = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
            dataset = Dataset(X=X, y=None, features=["f1", "f2", "f3"])

            vt = VarianceThreshold(threshold=0.2)
            vt._fit(dataset)
            new_dataset = vt._transform(dataset)

            assert new_dataset.X.shape[1]== 0
            assert new_dataset.features == []