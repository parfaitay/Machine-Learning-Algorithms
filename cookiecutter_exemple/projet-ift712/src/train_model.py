# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


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

    def KFoldCrossValidation(self, data, labels):
        data = np.asarray(data)
        labels = np.asarray(labels)
        kx_train = []
        kt_train = []
        kx_test = []
        kt_test = []
        sss = StratifiedShuffleSplit(10, test_size=0.2, random_state=0)
        sss.get_n_splits(data, labels)
        for train_index, test_index in sss.split(data, labels):
            #print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = data[train_index], data[test_index]
            t_train, t_test = labels[train_index], labels[test_index]
            kx_train.append(x_train)
            kt_train.append(t_train)
            kx_test.append(x_test)
            kt_test.append(t_train)

        return kx_train, kt_train, kx_test, kt_test

    def entrainement(self, data, labels, test, test_ids, classes):
        """
        Pour classifieur = 2:
            K-NN
            Les hyperparametres sont K(le nombre de voisins) et la metrique
        """

        # resultat du kfold cross validation
        # kx_train, kt_train, kx_test, kt_test = self.KFoldCrossValidation(data, labels)
        trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.2, random_state=0)

        if self.classifieur == 1:
            
            clf = LinearDiscriminantAnalysis(n_components=2)
            param ={}
            grid_lda= GridSearchCV(clf,param,cv=5)
           
            grid_lda.fit(trainData, trainLabels)
            accuracy = grid_lda.score(testData, testLabels)
            print("gridlda search accuracy: {:.2f}%".format(accuracy * 100))
            print("gridlda search best parameters: {}".format(grid_lda.best_params_))
            # print(classes)

        if self.classifieur == 2:
            # construct the set of hyperparameters to tune
            hparams = {"n_neighbors": np.arange(14, 31, 1), "metric": ["euclidean", "cityblock"]}
            # tune the hyperparameters via a cross-validated grid search
            print("tuning hyperparameters via grid search")
            clf = KNeighborsClassifier()
            # for x_train, t_train, x_test, t_test in kx_train, kt_train, kx_test, kt_test:
            # for i in range(len(kx_train)):
            #     x_train = kx_train[i]
            #     t_train = kt_train[i]
            #     x_test = kx_test[i]
            #     t_test = kt_test[i]
            grid = GridSearchCV(clf, hparams, cv=3);
            grid.fit(trainData, trainLabels)
            accuracy = grid.score(testData, testLabels)
            print("grid search accuracy: {:.2f}%".format(accuracy * 100))
            print("grid search best parameters: {}".format(grid.best_params_))
            test_predictions = grid.predict_proba(test)
            # Format DataFrame
            submission = pd.DataFrame(test_predictions, columns=classes)
            submission.insert(0, 'id', test_ids)
            submission.reset_index()

            # Export Submission
            submission.to_csv('submission.csv', index=False)
            submission.tail()

            #randomized search
            grid = RandomizedSearchCV(clf, hparams, cv=5)
            grid.fit(trainData, trainLabels)
            accuracy = grid.score(testData, testLabels)
            print("randomized search accuracy: {:.2f}%".format(accuracy * 100))
            print("randomized search best parameters: {}".format(grid.best_params_))

        if self.classifieur == 3:
            #penalite 
            #
            #des valeurs plus petites indiquent une régularisation plus forte pour le cas de C
            
            Logistic_Regression_params= {
                'penalty': ('l1', 'l2', ), 'C': [0.001, 0.01, 0.1, 1, 10,11,20, 100, ], }
            clf=LogisticRegression()

            param_search= GridSearchCV(clf, param_grid=Logistic_Regression_params,cv=10)
            param_search.fit(trainData,trainLabels)
            accuracy = param_search.score(testData, testLabels)
            print("regression logistic search accuracy: {:.2f}%".format(accuracy * 100))
            print("regression logistic search best parameters: {}".format(param_search.best_params_))

            test_predictions = param_search.predict_proba(test)
            # Format DataFrame
            submission = pd.DataFrame(test_predictions, columns=classes)
            submission.insert(0, 'id', test_ids)
            submission.reset_index()

            # Export Submission
            submission.to_csv('submissionteny2.csv', index=False)
            submission.tail()




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
