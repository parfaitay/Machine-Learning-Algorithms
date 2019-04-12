# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.svm import SVC


class trainModel:
    def __init__(self, classifieur=1):
        """
        Classification algorithms
        ``classifier`` :   1 for classification generative
                        2 for KNN
                        3 for SVM with three kernel polynomial, sigmoidal and rbf
                        4 for randomforest



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


        # Now we can compare the dimensions of the training set before and after applying PCA and see if we
        # managed to reduce the number of features.
        trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.2, random_state=0)

        if self.classifieur == 1:

            svc_sigmoid=SVC(probability=True, kernel='sigmoid',gamma='auto')
            AdaBoost_params = {
                    'n_estimators': [50, 100],
                    'learning_rate' : [0.01,0.05,0.1,0.3,1],
                             }
            
            AdaBoost_search = GridSearchCV(AdaBoostClassifier(algorithm='SAMME', base_estimator=svc_sigmoid),
                     param_grid = AdaBoost_params, cv=3)
                        
            AdaBoost_search.fit(trainData, trainLabels)
                

            AdaBoost_accuracy = AdaBoost_search.score(testData, testLabels)
            print("AdaBoost_search   accuracy: {:.2f}%".format(AdaBoost_accuracy * 100))
            print("AdaBoost_search search best parameters: {}".format(AdaBoost_search.best_params_))
        
                
        if self.classifieur == 2:
      

            # construct the set of hyperparameters to tune
            k_range = np.arange(1, 31, 1)
            hparams = {"n_neighbors": k_range, "metric": ["euclidean", "cityblock"]}
            # tune the hyperparameters via a cross-validated grid search
            print("tuning hyperparameters via grid search")
            clf = KNeighborsClassifier()
            # for x_train, t_train, x_test, t_test in kx_train, kt_train, kx_test, kt_test:
            # for i in range(len(kx_train)):
            #     x_train = kx_train[i]
            #     t_train = kt_train[i]
            #     x_test = kx_test[i]
            #     t_test = kt_test[i]
            grid = GridSearchCV(clf, hparams, cv=4, scoring='accuracy')
            grid.fit(trainData, trainLabels)
            accuracy = grid.score(testData, testLabels)
            # print("len1",len(testData))
            # print("len2",len(testLabels))
            # testpred = grid.predict_proba(testData)
            # ll = log_loss(testLabels, testpred)
            # print("log loss:", ll)
            print("KNN grid search accuracy: {:.2f}%".format(accuracy * 100))
            print("KNN grid search best parameters: {}".format(grid.best_params_))

            print(grid.cv_results_.keys())
            # create a list of the mean scores only
            # list comprehension to loop through grid.grid_scores
            grid_mean_scores = [result for result in grid.cv_results_['mean_test_score']]
            # print(grid_mean_scores)
            # plot the results
            plt.plot(k_range, grid_mean_scores)
            plt.xlabel('Value of K for KNN')
            plt.ylabel('Cross-Validated Accuracy')

            # kaggle
            test_predictions = grid.predict_proba(test)
            # Format DataFrame
            submission = pd.DataFrame(test_predictions, columns=classes)
            submission.insert(0, 'id', test_ids)
            submission.reset_index()

            # Export Submission
            submission.to_csv('submission.csv', index=False)
            submission.tail()

            #randomized search
            # grid = RandomizedSearchCV(clf, hparams, cv=5)
            # grid.fit(trainData, trainLabels)
            # accuracy = grid.score(testData, testLabels)
            # print("randomized search accuracy: {:.2f}%".format(accuracy * 100))
            # print("randomized search best parameters: {}".format(grid.best_params_))

        if self.classifieur == 3:
            # Cs = [0.01, 0.1, 1, 10, 100]
            # gammas = [0.001, 0.01, 0.1, 1, 10, 100]
            # hparams = {'C': Cs, 'gamma': gammas}
            # print("tuning SVM(kernel=rbf) hyperparameters via grid search")
            # clf = svm.SVC(kernel='rbf', probability=True)
            # grid_search = GridSearchCV(clf, hparams, cv=4)
            # grid_search.fit(trainData, trainLabels)
            # accuracy = grid_search.score(testData, testLabels)
            # print("SVM grid search accuracy: {:.2f}%".format(accuracy * 100))
            # print("SVM grid search best parameters: {}".format(grid_search.best_params_))
            # test_predictions = grid_search.predict_proba(test)
            # # Format DataFrame
            # submission = pd.DataFrame(test_predictions, columns=classes)
            # submission.insert(0, 'id', test_ids)
            # submission.reset_index()
            #
            # # Export Submission
            # submission.to_csv('submission.csv', index=False)
            # submission.tail()

            # Set the parameters by cross-validation
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

                clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                                   scoring='%s_macro' % score)
                clf.fit(trainData, trainLabels)

                print("Best parameters set found on development set:")
                print()
                print(clf.best_params_)
                print()
                print("Grid scores on development set:")
                print()
                means = clf.cv_results_['mean_test_score']
                stds = clf.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                    print("%0.3f (+/-%0.03f) for %r"
                          % (mean, std * 2, params))
                print()

            #    print("Detailed classification report:")
            #    print()
            #    print("The model is trained on the full development set.")
            #    print("The scores are computed on the full evaluation set.")
            #    print()
            #    y_true, y_pred = y_test, clf.predict(X_test)
            #    print(classification_report(y_true, y_pred))
            #    print()

        if self.classifieur == 6:
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

        if self.classifieur == 4:

            Perceptron_params= {
                'penalty': ('l1', 'l2', ),
                'alpha': [0.0001,0.1, 0.01, ],
                'n_iter': [1, 2, 5, 10, 100, 500, ], }

            clf=Perceptron()

            perceptron_search= GridSearchCV(clf, param_grid=Perceptron_params,cv=5)
            perceptron_search.fit(trainData,trainLabels)
            perceptron_accuracy = perceptron_search.score(testData, testLabels)
            print("perceptron  search accuracy: {:.2f}%".format(perceptron_accuracy * 100))
            print("perceptron logistic search best parameters: {}".format(perceptron_search.best_params_))


        if self.classifieur == 5:
    
            DecisionTree_params=  {
                'max_depth': (5, 10, 20, 50, 100, 500,999 ),
                'max_features': range(
                    len(trainData[0]) - 15, len(trainData[0])), }

            clf = tree.DecisionTreeClassifier()

            DecisionTree_search= GridSearchCV(clf, param_grid=DecisionTree_params,cv=5)
            DecisionTree_search.fit(trainData,trainLabels)

            


           
            DecisionTree_accuracy = DecisionTree_search.score(testData, testLabels)

            for  result in DecisionTree_search.cv_results_ :
                print(" parametre  "+  result['grid_mean_scores'])
            #plt.plot(,grid_mean_scores)

            print("DecisionTree  search accuracy: {:.2f}%".format(DecisionTree_accuracy * 100))
            print("DecisionTree  search best parameters: {}".format(DecisionTree_search.best_params_))


        if self.classifieur == 6:
        
            RandomForest_params= {
                'max_depth': (5, 10, 20,50, 100, 500,999 ),
                'n_estimators': (10,50, 100, ),
                'max_features': range(
                    len(trainData[0]) - 15 , len(trainData[0])), }

            clf= RandomForestClassifier()
        
            RandomForest_search= GridSearchCV(clf, param_grid=RandomForest_params,cv=5)
            RandomForest_search.fit(trainData,trainLabels)
            RandomForest_accuracy = RandomForest_search.score(testData, testLabels)
            print("perceptron  search accuracy: {:.2f}%".format(RandomForest_accuracy * 100))
            print("perceptron logistic search best parameters: {}".format(RandomForest_search.best_params_))

