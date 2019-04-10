# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class MakeDataset:
    def __init__(self):
        self.train = pd.read_csv('../data/train.csv')
        self.test = pd.read_csv('../data/test.csv')

    def prepare_data(self):
        """
        to-do
        """
        le = LabelEncoder().fit(self.train.species)
        labels = le.transform(self.train.species)
        classes = le.classes_
        data = self.train.drop(['species', 'id'], axis=1)
        test_ids = self.test.id
        test = self.test.drop(['id'], axis=1)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        test = scaler.transform(test)
        return classes, data, labels, test, test_ids


    
    

    def print_data(self, x, t, scatter=True):
        """
        to-do
        """


