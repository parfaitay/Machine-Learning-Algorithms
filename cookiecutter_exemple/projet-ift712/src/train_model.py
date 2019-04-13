# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC


class trainModel:
    def __init__(self, classifieur=1):
        """
        Classification algorithms
        ``classifier`` :    1 for Adaboost with linear Kernel
                            2 for KNN
                            3 for SVM with three kernel polynomial, sigmoidal and rbf
                            4 for Logistic Regression
                            5 Perceptron
                            6 RandomForest
        """
        self.classifieur = classifieur

    def entrainement(self, trainData, trainLabels, testData, testLabels, test, classes, test_ids):
        """
        Train the models with the train dataset
        :param trainData:
        :param trainLabels:
        :param testData:
        :param testLabels:
        :return:
        """

        if self.classifieur == 1:
            svc_linear = SVC(probability=True, kernel='linear', gamma='auto')
            AdaBoost_params = {
                'n_estimators': [5, 10, 15, 20],
                'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
            }
            clf = AdaBoostClassifier(algorithm='SAMME', base_estimator=svc_linear)

            AdaBoost_search = GridSearchCV(clf, param_grid=AdaBoost_params, cv=3)

            AdaBoost_search.fit(trainData, trainLabels)

            AdaBoost_accuracy = AdaBoost_search.score(testData, testLabels)
            print("AdaBoost_search   accuracy: {:.2f}%".format(AdaBoost_accuracy * 100))
            print("AdaBoost_search search best parameters: {}".format(AdaBoost_search.best_params_))

        if self.classifieur == 2:
            # construct the set of hyperparameters to tune
            k_range = np.arange(2, 31, 1)
            hparams = {"n_neighbors": k_range, "metric": ["euclidean", "cityblock"]}
            # tune the hyperparameters via a cross-validated grid search
            print("tuning hyperparameters via grid search")
            clf = KNeighborsClassifier()
            grid = GridSearchCV(clf, hparams, cv=3, scoring='accuracy')
            grid.fit(trainData, trainLabels)
            knn_accuracy = grid.score(testData, testLabels)
            print("KNN grid search accuracy: {:.2f}%".format(knn_accuracy * 100))
            print("KNN grid search best parameters: {}".format(grid.best_params_))
            # kaggle
            test_predictions = grid.predict_proba(test)
            # Format DataFrame
            submission = pd.DataFrame(test_predictions, columns=classes)
            submission.insert(0, 'id', test_ids)
            submission.reset_index()

            # Export Submission
            submission.to_csv('submission.csv', index=False)
            submission.tail()

        if self.classifieur == 3:
            # Set the parameters for cross-validation
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                                 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                                {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                                 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                                {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                                ]

            scores = ['precision', 'recall']

            for score in scores:
                print("# Tuning hyper-parameters for %s" % score)
                print()

                clf = GridSearchCV(SVC(C=1, probability=True), tuned_parameters, cv=3,
                                   scoring='%s_macro' % score)
                clf.fit(trainData, trainLabels)
                svm_accuracy = clf.score(testData, testLabels)
                print("SVM grid search accuracy: {:.2f}%".format(svm_accuracy * 100))
                print("SVM grid search best parameters: {}".format(clf.best_params_))
                # kaggle
                test_predictions = clf.predict_proba(test)
                # Format DataFrame
                submission = pd.DataFrame(test_predictions, columns=classes)
                submission.insert(0, 'id', test_ids)
                submission.reset_index()

                # Export Submission
                submission.to_csv('submission.csv', index=False)
                submission.tail()

        if self.classifieur == 4:
            Logistic_Regression_params= {
                'penalty': ('l1', 'l2', ), 'C': [0.001, 0.01, 0.1, 1, 10, 11, 20, 100, ], }
            clf = LogisticRegression()

            param_search = GridSearchCV(clf, param_grid=Logistic_Regression_params, cv=3)
            param_search.fit(trainData,trainLabels)
            accuracy = param_search.score(testData, testLabels)
            print("regression logistic search accuracy: {:.2f}%".format(accuracy * 100))
            print("regression logistic search best parameters: {}".format(param_search.best_params_))

        if self.classifieur == 5:

            Perceptron_params = {
                'penalty': ('l1', 'l2', ),
                'alpha': [0.0001, 0.1, 0.01, ],
                'max_iter': [100, 500, 700], 'tol': [1e-3]}

            clf = Perceptron()

            perceptron_search = GridSearchCV(clf, param_grid=Perceptron_params, cv=3)
            perceptron_search.fit(trainData,trainLabels)
            perceptron_accuracy = perceptron_search.score(testData, testLabels)
            print("perceptron search accuracy: {:.2f}%".format(perceptron_accuracy * 100))
            print("perceptron search best parameters: {}".format(perceptron_search.best_params_))

        if self.classifieur == 6:
        
            RandomForest_params= {
                'max_depth': (5, 10, 20, 50, 100, 1000),
                'n_estimators': (10, 50, 100)}

            clf = RandomForestClassifier()
        
            RandomForest_search= RandomizedSearchCV(clf, RandomForest_params, cv=3)
            RandomForest_search.fit(trainData, trainLabels)
            RandomForest_accuracy = RandomForest_search.score(testData, testLabels)
            print("RandomForest  search accuracy: {:.2f}%".format(RandomForest_accuracy * 100))
            print("RandomForest logistic search best parameters: {}".format(RandomForest_search.best_params_))

