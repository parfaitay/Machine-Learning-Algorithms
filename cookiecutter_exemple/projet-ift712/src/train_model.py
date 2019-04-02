# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



class trainModel:
    def __init__(self, classifieur=1):
        """
        Algorithmes de classification lineaire

        L'argument ``lamb`` est une constante pour régulariser la magnitude
        des poids w et w_0

        ``methode`` :   1 pour classification generative
                        2 pour k plus proches voisins
                        3 pour Perceptron sklearn
                        4 pour arbre de decision



        """
        self.classifieur = classifieur
        

    def entrainement(self, x_train, t_train, classes):
        """
        """
        if self.classifieur == 1:
            clf = LinearDiscriminantAnalysis()
            clf.fit(x_train, t_train)
            LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001)
            print(clf.predict(x_train))
            print(classes)

             

    def prediction(self, x):
        """
        """
        return 0

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        on calcule lerreur de la prediction
        """
 
        err = (t - prediction) ** 2
        return err
