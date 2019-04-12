# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np


class MakeDataset:
    def __init__(self):
        self.train = pd.read_csv('../data/train.csv')
        self.test = pd.read_csv('../data/test.csv')

    def prepare_data(self):
        """
        perform the encoding of the label
        return data, label, classe's names test and test ids(for submission on kaggle)
        """
        le = LabelEncoder().fit(self.train.species)
        labels = le.transform(self.train.species)
        classes = le.classes_
        data = self.train.drop(['species', 'id'], axis=1)
        test_ids = self.test.id
        test = self.test.drop(['id'], axis=1)
        return classes, data, labels, test, test_ids

    def normalizer(self, data):
        """
        Normalize the data (standardscaler from sklearn)
        return standardadized data
        """
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        return data

    
    

    def apply_pca(self, data):
        """
        Perform the principal analysis component PCA on data given a number of components to keep
        """
        pca = PCA(n_components=0.90, svd_solver='full')
        # Then we fit pca on our training set and we apply to the same entire set
        data_pca = pca.fit_transform(data)
        return data_pca

    def pearson(self, x, y):
        if len(x) != len(y):
            print("I can't calculate Pearson's Coefficient, sets are not of the same length!")
            return
        else:
            sumxy = 0
            for i in range(len(x)):
                sumxy = sumxy + x[i] * y[i]
            pearson = (sumxy - len(x) * np.mean(x) * np.mean(y)) / ((len(x) - 1) * np.std(x, ddof=1) * np.std(y, ddof=1))
            return pearson
