# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Visualize:
    def __init__(self):
        self.train = pd.read_csv('../data/train.csv')
        self.test = pd.read_csv('../data/test.csv')

    def print_data_perClass(self):
        """
        to-do
        """
        width = 12
        height = 20
        plt.figure(figsize=(width, height))
        sns.countplot(y=self.train['species'], label="Count")
        plt.show()

    def print_histogram(self):
        """
        to-do
        """
        import pylab as pl
        self.train.drop('species', axis=1).drop('id', axis=1).hist(bins=30, figsize=(30, 40))
        pl.suptitle("Histogram for each numeric input variable")
        plt.show()


