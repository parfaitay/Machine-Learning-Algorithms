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


def main():

    if len(sys.argv) > 7:
        usage = "\n Usage: python mainly_class.py classifier_type normalize_data _apply_pca_data visualize \
        \n\n\t classifier_type : 1 => Classification generative\
        \n\t classifier_type : 2 => Perceptron + SDG \n\t method : 3 => Perceptron + SDG [sklearn]\
        \n\t nb_train, nb_test : nombre de donnees d'entrainement et de test\
        \n\t lambda >=0\
        \n\t bruit : multiplicateur de la matrice de variance-covariance (entre 0.1 et 50)\
        \n\t don_ab : production ou non de données aberrantes (0 ou 1) \
        \n\n\t ex : python classifieur_lineaire.py 1 280 280 0.001 1 1"
        print(usage)
        return

    classifier_type = int(sys.argv[1])
    normalize = int(sys.argv[2])
    pca = int(sys.argv[3])
    vis = float(sys.argv[4])
    #bruit = float(sys.argv[5])
    #donnees_aberrantes = bool(int(sys.argv[6]))

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

    if classifier_type != 0:
        print(" Training with the specified classifier...")
        train_model = tm.trainModel(classifieur=classifier_type)
        train_model.entrainement(data, labels, test, test_ids, classes)
    
    # err_train = 50
    # err_test = 50

    # print('Erreur train = ', err_train, '%')
    # print('Erreur test = ', err_test, '%')
    #analyse_erreur(err_train, err_test)
    
    


if __name__ == "__main__":
    main()
