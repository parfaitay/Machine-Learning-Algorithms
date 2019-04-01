# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class MakeDataset:
    def __init__(self):
        self.train = pd.read_csv('../data/raw/train.csv')
        self.test = pd.read_csv('../data/raw/test.csv')

    def prepare_data(self):
        """
        to-do
        """
        le = LabelEncoder().fit(self.train.species)
        t_train = le.transform(self.train.species)
        classes = le.classes_
        x_train = self.train.drop(['species', 'id'], axis=1)
        return classes, x_train, t_train,

    def print_data(self, x, t, scatter=True):
        """
        to-do
        """

