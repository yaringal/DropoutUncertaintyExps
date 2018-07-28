# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his 
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

# This file contains code to train dropout networks on the UCI datasets using the following algorithm:
# 1. Create 20 random splits of the training-test dataset.
# 2. For each split:
# 3.   Create a validation (val) set taking 20% of the training set.
# 4.   Get best hyperparameters: dropout_rate and tau by training on (train-val) set and testing on val set.
# 5.   Train a network on the entire training set with the best pair of hyperparameters.
# 6.   Get the performance (MC RMSE and log-likelihood) on the test set.
# 7. Report the averaged performance (Monte Carlo RMSE and log-likelihood) on all 20 splits.

import math
import numpy as np
import argparse
import sys

parser=argparse.ArgumentParser()

parser.add_argument('--dir', '-d', required=True, help='Name of the UCI Dataset directory. Eg: bostonHousing')
parser.add_argument('--epochx','-e', default=500, type=int, help='Multiplier for the number of epochs for training.')
parser.add_argument('--hidden', '-nh', default=2, type=int, help='Number of hidden layers for the neural net')

args=parser.parse_args()

data_directory = args.dir
epochs_multiplier = args.epochx
num_hidden_layers = args.hidden

sys.path.append('net/')

import net

# We delete previous results

from subprocess import call


_RESULTS_VALIDATION_LL = "./UCI_Datasets/" + data_directory + "/results/validation_ll_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_VALIDATION_RMSE = "./UCI_Datasets/" + data_directory + "/results/validation_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_VALIDATION_MC_RMSE = "./UCI_Datasets/" + data_directory + "/results/validation_MC_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"

_RESULTS_TEST_LL = "./UCI_Datasets/" + data_directory + "/results/test_ll_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_TAU = "./UCI_Datasets/" + data_directory + "/results/test_tau_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_RMSE = "./UCI_Datasets/" + data_directory + "/results/test_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_MC_RMSE = "./UCI_Datasets/" + data_directory + "/results/test_MC_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_LOG = "./UCI_Datasets/" + data_directory + "/results/log_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"

_DATA_DIRECTORY_PATH = "./UCI_Datasets/" + data_directory + "/data/"
_DROPOUT_RATES_FILE = _DATA_DIRECTORY_PATH + "dropout_rates.txt"
_TAU_VALUES_FILE = _DATA_DIRECTORY_PATH + "tau_values.txt"
_DATA_FILE = _DATA_DIRECTORY_PATH + "data.txt"
_HIDDEN_UNITS_FILE = _DATA_DIRECTORY_PATH + "n_hidden.txt"
_EPOCHS_FILE = _DATA_DIRECTORY_PATH + "n_epochs.txt"
_INDEX_FEATURES_FILE = _DATA_DIRECTORY_PATH + "index_features.txt"
_INDEX_TARGET_FILE = _DATA_DIRECTORY_PATH + "index_target.txt"
_N_SPLITS_FILE = _DATA_DIRECTORY_PATH + "n_splits.txt"

def _get_index_train_test_path(split_num, train = True):
    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """
    if train:
        return _DATA_DIRECTORY_PATH + "index_train_" + str(split_num) + ".txt"
    else:
        return _DATA_DIRECTORY_PATH + "index_test_" + str(split_num) + ".txt" 


print ("Removing existing result files...")
call(["rm", _RESULTS_VALIDATION_LL])
call(["rm", _RESULTS_VALIDATION_RMSE])
call(["rm", _RESULTS_VALIDATION_MC_RMSE])
call(["rm", _RESULTS_TEST_LL])
call(["rm", _RESULTS_TEST_TAU])
call(["rm", _RESULTS_TEST_RMSE])
call(["rm", _RESULTS_TEST_MC_RMSE])
call(["rm", _RESULTS_TEST_LOG])
print ("Result files removed.")

# We fix the random seed

np.random.seed(1)

print ("Loading data and other hyperparameters...")
# We load the data

data = np.loadtxt(_DATA_FILE)

# We load the number of hidden units

n_hidden = np.loadtxt(_HIDDEN_UNITS_FILE).tolist()

# We load the number of training epocs

n_epochs = np.loadtxt(_EPOCHS_FILE).tolist()

# We load the indexes for the features and for the target

index_features = np.loadtxt(_INDEX_FEATURES_FILE)
index_target = np.loadtxt(_INDEX_TARGET_FILE)

X = data[ : , [int(i) for i in index_features.tolist()] ]
y = data[ : , int(index_target.tolist()) ]

# We iterate over the training test splits

n_splits = np.loadtxt(_N_SPLITS_FILE)
print ("Done.")

errors, MC_errors, lls = [], [], []
for split in range(int(n_splits)):

    # We load the indexes of the training and test sets
    print ('Loading file: ' + _get_index_train_test_path(split, train=True))
    print ('Loading file: ' + _get_index_train_test_path(split, train=False))
    index_train = np.loadtxt(_get_index_train_test_path(split, train=True))
    index_test = np.loadtxt(_get_index_train_test_path(split, train=False))

    X_train = X[ [int(i) for i in index_train.tolist()] ]
    y_train = y[ [int(i) for i in index_train.tolist()] ]
    
    X_test = X[ [int(i) for i in index_test.tolist()] ]
    y_test = y[ [int(i) for i in index_test.tolist()] ]

    X_train_original = X_train
    y_train_original = y_train
    num_training_examples = int(0.8 * X_train.shape[0])
    X_validation = X_train[num_training_examples:, :]
    y_validation = y_train[num_training_examples:]
    X_train = X_train[0:num_training_examples, :]
    y_train = y_train[0:num_training_examples]
    
    # Printing the size of the training, validation and test sets
    print ('Number of training examples: ' + str(X_train.shape[0]))
    print ('Number of validation examples: ' + str(X_validation.shape[0]))
    print ('Number of test examples: ' + str(X_test.shape[0]))
    print ('Number of train_original examples: ' + str(X_train_original.shape[0]))

    # List of hyperparameters which we will try out using grid-search
    dropout_rates = np.loadtxt(_DROPOUT_RATES_FILE).tolist()
    tau_values = np.loadtxt(_TAU_VALUES_FILE).tolist()

    # We perform grid-search to select the best hyperparameters based on the highest log-likelihood value
    best_network = None
    best_ll = -float('inf')
    best_tau = 0
    best_dropout = 0
    for dropout_rate in dropout_rates:
        for tau in tau_values:
            print ('Grid search step: Tau: ' + str(tau) + ' Dropout rate: ' + str(dropout_rate))
            network = net.net(X_train, y_train, ([ int(n_hidden) ] * num_hidden_layers),
                    normalize = True, n_epochs = int(n_epochs * epochs_multiplier), tau = tau,
                    dropout = dropout_rate)

            # We obtain the test RMSE and the test ll from the validation sets

            error, MC_error, ll = network.predict(X_validation, y_validation)
            if (ll > best_ll):
                best_ll = ll
                best_network = network
                best_tau = tau
                best_dropout = dropout_rate
                print ('Best log_likelihood changed to: ' + str(best_ll))
                print ('Best tau changed to: ' + str(best_tau))
                print ('Best dropout rate changed to: ' + str(best_dropout))
            
            # Storing validation results
            with open(_RESULTS_VALIDATION_RMSE, "a") as myfile:
                myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                myfile.write(repr(error) + '\n')

            with open(_RESULTS_VALIDATION_MC_RMSE, "a") as myfile:
                myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                myfile.write(repr(MC_error) + '\n')

            with open(_RESULTS_VALIDATION_LL, "a") as myfile:
                myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                myfile.write(repr(ll) + '\n')

    # Storing test results
    best_network = net.net(X_train_original, y_train_original, ([ int(n_hidden) ] * num_hidden_layers),
                    normalize = True, n_epochs = int(n_epochs * epochs_multiplier), tau = best_tau,
                    dropout = best_dropout)
    error, MC_error, ll = best_network.predict(X_test, y_test)
    
    with open(_RESULTS_TEST_RMSE, "a") as myfile:
        myfile.write(repr(error) + '\n')

    with open(_RESULTS_TEST_MC_RMSE, "a") as myfile:
        myfile.write(repr(MC_error) + '\n')

    with open(_RESULTS_TEST_LL, "a") as myfile:
        myfile.write(repr(ll) + '\n')

    with open(_RESULTS_TEST_TAU, "a") as myfile:
        myfile.write(repr(best_network.tau) + '\n')

    print ("Tests on split " + str(split) + " complete.")
    errors += [error]
    MC_errors += [MC_error]
    lls += [ll]

with open(_RESULTS_TEST_LOG, "a") as myfile:
    myfile.write('errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(errors), np.std(errors), np.std(errors)/math.sqrt(n_splits),
        np.percentile(errors, 50), np.percentile(errors, 25), np.percentile(errors, 75)))
    myfile.write('MC errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(MC_errors), np.std(MC_errors), np.std(MC_errors)/math.sqrt(n_splits),
        np.percentile(MC_errors, 50), np.percentile(MC_errors, 25), np.percentile(MC_errors, 75)))
    myfile.write('lls %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(lls), np.std(lls), np.std(lls)/math.sqrt(n_splits), 
        np.percentile(lls, 50), np.percentile(lls, 25), np.percentile(lls, 75)))
