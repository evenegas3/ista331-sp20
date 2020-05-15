import unittest
from lab8 import *
from compare_pandas import compare_series

import pandas as pd
from datetime import datetime as dt, timedelta

''' File needed: compare_pandas.py '''

class TestAssignment2(unittest.TestCase):
    def setUp(self):
        self.fv1 = pd.Series([0, 19, 61000, 0.625], ['Marital Status', 'Age', 'Income', 'FTE'])
        self.fv1_scaled = pd.Series([0, 0.02, 0.340741, 0.625], ['Marital Status', 'Age', 'Income', 'FTE'])
        self.fv2 = pd.Series([35, 78000, 1.0, 2], ['Age', 'Income', 'FTE', 'Num Pets'])
        self.fv2_scaled = pd.Series([0.34, 0.466667, 1.0, 0.2], ['Age', 'Income', 'FTE', 'Num Pets'])
        self.fv3 = pd.Series([18, 15000, 0.0], ['Age', 'Income', 'FTE'])
        self.fv4 = pd.Series([68, 150000, 1.0], ['Age', 'Income', 'FTE'])
        self.fv5 = pd.Series([68, 150000, 1.0], ['Age2', 'Income2', 'FTE2'])
        self.zero = pd.Series([0, 0, 0.0], ['Age', 'Income', 'FTE'])
        self.one = pd.Series([1, 1, 1.0], ['Age', 'Income', 'FTE'])
        self.lower = pd.Series([0, 18, 15000, 0.0, 0], ['Marital Status', 'Age', 'Income', 'FTE', 'Num Pets'])
        self.upper = pd.Series([1, 68, 150000, 1.0, 10], ['Marital Status', 'Age', 'Income', 'FTE', 'Num Pets'])
        index1 = [dt(2017, 10, 10), dt(2017, 11, 10), dt(2017, 12, 10), dt(2018, 1, 10), dt(2018, 2, 10)]
        data1 = [3, 1, -5, 1, 3]
        self.ts1 = pd.Series(data1, index1)
        index2 = [dt(1989, 5, 10), dt(2000, 11, 3), dt(2017, 12, 10), dt(2018, 1, 5), 
            dt(2018, 2, 10), dt(2019, 2, 10), dt(2021, 2, 11)]
        data2 = [3, 1, -5, 1, 3, 77, 88]
        self.ts2 = pd.Series(data2, index2)
    
    def test_euclidean_distance(self):
        # Note how the big number feature (Income) overwhelm the smaller features
        # Can't use assertEqual - need to handle roundoff error
        self.assertTrue(abs(euclidean_distance(self.fv1, self.fv2) - 17000.00753355) < 0.00001)
        self.assertTrue(abs(euclidean_distance(self.fv1, self.fv1)) < 0.00001)
        self.assertTrue(abs(euclidean_distance(self.fv3, self.fv4) - 135000.009263) < 0.00001)
        self.assertTrue(pd.isnull(euclidean_distance(self.fv5, self.fv4)))
        
    def test_scaled_feature_vector(self):
        fv1_copy = self.fv1.copy()
        self.assertTrue(compare_series(self.fv1_scaled, scaled_feature_vector(self.fv1, self.lower, self.upper)))
        self.assertTrue(compare_series(fv1_copy, self.fv1))
        self.assertFalse(compare_series(self.fv1_scaled, scaled_feature_vector(self.fv2, self.lower, self.upper)))
        self.assertTrue(compare_series(self.zero, scaled_feature_vector(self.fv3, self.lower, self.upper)))
        self.assertTrue(compare_series(self.one, scaled_feature_vector(self.fv4, self.lower, self.upper)))
        
    def test_scaled_euclidean_distance(self):
        dist, n = scaled_euclidean_distance(self.fv1, self.fv2, self.lower, self.upper)
        self.assertTrue(abs(0.5088048 - dist) < 0.00001)
        self.assertEqual(3, n)
        dist, n = scaled_euclidean_distance(self.fv3, self.fv4, self.lower, self.upper)
        self.assertTrue(abs(1.7320508 - dist) < 0.00001)
        self.assertEqual(3, n)
        
    def test_similarity(self):
        self.assertTrue(abs(similarity(self.fv3, self.fv4, self.lower, self.upper) < 0.00001))
        self.assertTrue(abs(0.28454 - similarity(self.fv1, self.fv4, self.lower, self.upper) < 0.00001))
        self.assertTrue(abs(1 - similarity(self.fv1, self.fv1, self.lower, self.upper) < 0.00001))


test = unittest.defaultTestLoader.loadTestsFromTestCase(TestAssignment2)
results = unittest.TextTestRunner().run(test)
print('Correctness score = ', str((results.testsRun - len(results.errors) - len(results.failures)) / results.testsRun * 100) + ' / 100')
