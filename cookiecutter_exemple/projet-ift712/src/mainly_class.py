# -*- coding: utf-8 -*-

import sys
import train_model as tm
import make_dataset as md
import visualize as vz

#################################################
# Execution as a script in a terminal
#
# Example:
# python mainly_class.py 1 1 0 0
#
#################################################
from sklearn.model_selection import train_test_split


def main():

    if len(sys.argv) < 5:
        usage = "\n Usage: python mainly_class.py classifier_type normalize_data apply_pca_data visualize run_all\
        \n\n\t classifier_type : 1 => Adaboost\
        \n\t classifier_type : 2 => KNN \n\t classifier_type : 3 => SVM \n\t classifier_type : 4 => logistic regression\
        \n\t classifier_type : 5 => Perceptron\n\t classifier_type : 6 => RandomForest\
        \n\t normalize_data: Normalize the training dataset\
        \n\t apply_pca_data: Apply pca on the training dataset\
        \n\t visualize: Visualize some data of the dataset\
        \n\t run_all: Run all models\
        \n\n\t ex : python mainly_class.py 0 0 0 0 1"
        print(usage)
        return

    classifier_type = int(sys.argv[1])
    normalize = int(sys.argv[2])
    pca = int(sys.argv[3])
    vis = int(sys.argv[4])
    run_all = int(sys.argv[5])

    print("Making Dataset...")
    generateur_donnees = md.MakeDataset()
    classes, data, labels, test, test_ids = generateur_donnees.prepare_data()

    if normalize == 1:
        print("Normalizing Dataset...")
        data = generateur_donnees.normalizer(data)

    if pca == 1:
        print("Applying PCA on dataset...")
        data = generateur_donnees.apply_pca(data)

    # Visualizing Data
    if vis == 1:
        visio = vz.Visualize()
        visio.show_correlation(data)

    # Run for each classifier
    if classifier_type != 0:
        print(" Training with the specified classifier...")
        train_model = tm.trainModel(classifieur=classifier_type)
        trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.2, random_state=0)
        train_model.entrainement(trainData, trainLabels, testData, testLabels, test, classes, test_ids)

    # Run for all classifiers
    if run_all == 1:
        print(" Training with all classifiers...")
        trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.2, random_state=0)
        train_model = tm.trainModel(classifieur=1)
        train_model.entrainement(trainData, trainLabels, testData, testLabels)
        train_model = tm.trainModel(classifieur=2)
        train_model.entrainement(trainData, trainLabels, testData, testLabels)
        train_model = tm.trainModel(classifieur=3)
        train_model.entrainement(trainData, trainLabels, testData, testLabels)
        train_model = tm.trainModel(classifieur=4)
        train_model.entrainement(trainData, trainLabels, testData, testLabels)
        train_model = tm.trainModel(classifieur=5)
        train_model.entrainement(trainData, trainLabels, testData, testLabels)
        train_model = tm.trainModel(classifieur=6)
        train_model.entrainement(trainData, trainLabels, testData, testLabels)


if __name__ == "__main__":
    main()
