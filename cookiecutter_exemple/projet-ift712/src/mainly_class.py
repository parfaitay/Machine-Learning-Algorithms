# -*- coding: utf-8 -*-

import sys
import train_model as tm
import make_dataset as md
import visualize as vz

#################################################
# Execution en tant que script dans un terminal
#
# Exemple:
# python classifieur_lineaire.py 1 280 280 0.001 1 1
#
#################################################


def main():

    if len(sys.argv) > 7:
        usage = "\n Usage: python classifieur.py method nb_train nb_test lambda bruit corruption don_ab\
        \n\n\t method : 1 => Classification generative\
        \n\t method : 2 => Perceptron + SDG \n\t method : 3 => Perceptron + SDG [sklearn]\
        \n\t nb_train, nb_test : nombre de donnees d'entrainement et de test\
        \n\t lambda >=0\
        \n\t bruit : multiplicateur de la matrice de variance-covariance (entre 0.1 et 50)\
        \n\t don_ab : production ou non de données aberrantes (0 ou 1) \
        \n\n\t ex : python classifieur_lineaire.py 1 280 280 0.001 1 1"
        print(usage)
        return

    type_classifieur = int(sys.argv[1])
    #nb_train = int(sys.argv[2])
    #nb_test = int(sys.argv[3])
    #lamb = float(sys.argv[4])
    #bruit = float(sys.argv[5])
    #donnees_aberrantes = bool(int(sys.argv[6]))

    print("Generation des données d'entrainement...")
    
    generateur_donnees = md.MakeDataset()
    classes, data, labels, test, test_ids = generateur_donnees.prepare_data()

    print(" entrainement...")
    # print(" classes...",classes)
    # On entraine le modèle

    train_model = tm.trainModel(classifieur=type_classifieur)
    train_model.entrainement(data, labels, test, test_ids, classes)
    
    err_train = 50
    err_test = 50

    print('Erreur train = ', err_train, '%')
    print('Erreur test = ', err_test, '%')
    #analyse_erreur(err_train, err_test)

    # Affichage
    # visio = vz.Visualize()
    # visio.print_histogram()



if __name__ == "__main__":
    main()
