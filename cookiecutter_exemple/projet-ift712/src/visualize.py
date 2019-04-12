# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import matplotlib.cm as cm

import make_dataset as m


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

    def scatter_matrix(self, data):
        feature_names = ['shape1', 'shape2', 'shape3', 'shape4', 'shape5', 'shape6', 'shape7', 'shape8', 'shape9']
        X = data[feature_names]
        y = data['species']
        cmap = cm.get_cmap('gnuplot')
        scatter = pd.plotting.scatter_matrix(X, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(30, 40), cmap=cmap)
        plt.suptitle('Scatter-matrix for each input variable')

    def show_correlation(self, traindf):

        md = m.MakeDataset()
        # First we find the sets of margin, shape and texture columns
        margin_cols = [col for col in traindf.columns if 'margin' in col]
        shape_cols = [col for col in traindf.columns if 'shape' in col]
        texture_cols = [col for col in traindf.columns if 'texture' in col]
        margin_pear, shape_pear, texture_pear = [], [], []

        # Then we calculate the correlation coefficients for each couple of columns: we can either do this
        # between random columns of between consecutive columns, the difference won't matter much since we are
        # just exploring the data
        for i in range(len(margin_cols) - 1):
            margin_pear.append(md.pearson(traindf[margin_cols[i]], traindf[margin_cols[i + 1]]))
        # margin_pear.append(pearson(traindf[margin_cols[randint(0,len(margin_cols)-1)]],\
        # traindf[margin_cols[randint(0,len(margin_cols)-1)]]))
        for i in range(len(shape_cols) - 1):
            shape_pear.append(md.pearson(traindf[shape_cols[i]], traindf[shape_cols[i + 1]]))
        # shape_pear.append(pearson(traindf[shape_cols[randint(0,len(shape_cols)-1)]],\
        # traindf[shape_cols[randint(0,len(shape_cols)-1)]]))
        for i in range(len(texture_cols) - 1):
            texture_pear.append(md.pearson(traindf[texture_cols[i]], traindf[texture_cols[i + 1]]))
        # texture_pear.append(pearson(traindf[texture_cols[randint(0,len(texture_cols)-1)]],\
        # traindf[texture_cols[randint(0,len(texture_cols)-1)]]))

        # We calculate average and standard deviation for each cathergory
        # and we give it a position on the X axis of the graph
        margin_mean, margin_std = np.mean(margin_pear), np.std(margin_pear, ddof=1)
        margin_x = [0] * len(margin_pear)
        shape_mean, shape_std = np.mean(shape_pear), np.std(shape_pear, ddof=1)
        shape_x = [1] * len(shape_pear)
        texture_mean, texture_std = np.mean(texture_pear), np.std(texture_pear, ddof=1)
        texture_x = [2] * len(texture_pear)

        # We set up the graph
        fig = plt.figure(figsize=(8, 6))
        gs1 = gridspec.GridSpec(1, 2)  # , height_ratios=[1, 1])
        ax1, ax2 = fig.add_subplot(gs1[0]), fig.add_subplot(gs1[1])
        ax1.margins(0.05), ax2.margins(0.05)

        # We fill the first graph with a scatter plot on a single axis for each category and we add
        # mean and standard deviation, which we can also print to screen as a reference
        ax1.scatter(margin_x, margin_pear, color='blue', alpha=.3, s=100)
        ax1.errorbar([0], margin_mean, yerr=margin_std, color='white', alpha=1, fmt='o', mec='white', lw=2)
        ax1.scatter(shape_x, shape_pear, color='red', alpha=.3, s=100)
        ax1.errorbar([1], shape_mean, yerr=shape_std, color='white', alpha=1, fmt='o', mec='white', lw=2)
        ax1.scatter(texture_x, texture_pear, color='green', alpha=.3, s=100)
        ax1.errorbar([2], texture_mean, yerr=texture_std, color='white', alpha=1, fmt='o', mec='white', lw=2)
        ax1.set_ylim(-1.25, 1.25), ax1.set_xlim(-0.25, 2.25)
        ax1.set_xticks([0, 1, 2]), ax1.set_xticklabels(['margin', 'shape', 'texture'], rotation='vertical')
        ax1.set_xlabel('Category'), ax1.set_ylabel('Pearson\'s Correlation')
        ax1.set_title('Neighbours Correlation')
        ax1.set_aspect(2.5)

        print("Pearson's Correlation between neighbours\n==========================================")
        print("Margin: " + '{:1.3f}'.format(margin_mean) + u' \u00B1 ' \
              + '{:1.3f}'.format(margin_std))
        print("Shape: " + '{:1.3f}'.format(shape_mean) + u' \u00B1 ' \
              + '{:1.3f}'.format(shape_std))
        print("Texture: " + '{:1.3f}'.format(texture_mean) + u' \u00B1 ' \
              + '{:1.3f}'.format(texture_std))

        # And now, we build a more detailed (and expensive!) correlation matrix,
        # but only for the shape category, which, as we will see, is highly correlated
        shape_mat = []

        for i in range(traindf[shape_cols].shape[1]):
            shape_mat.append([])
            for j in range(traindf[shape_cols].shape[1]):
                shape_mat[i].append(md.pearson(traindf[shape_cols[i]], traindf[shape_cols[j]]))

        cmap = cm.RdBu_r
        MS = ax2.imshow(shape_mat, interpolation='none', cmap=cmap, vmin=-1, vmax=1)
        ax2.set_xlabel('Shape Feature'), ax2.set_ylabel('Shape Feature')
        cbar = plt.colorbar(MS, ticks=np.arange(-1.0, 1.1, 0.2))
        cbar.set_label('Pearson\'s Correlation')
        ax2.set_title('Shape Category Correlation Matrix')

        # And we have a look at the resulting graphs
        gs1.tight_layout(fig)
        plt.show()
