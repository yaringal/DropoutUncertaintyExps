# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his 
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

# This experiment uses the optimal model precision tau obtained from experiment_BO.py and 
# runs the model with a larger number of iterations.

import math
import numpy as np
import sys

if (len(sys.argv) != 2):
    print ('Dataset directory expected as parameter.')
    print ('Example format: python experiment_2_layers.py bostonHousing')
    exit(1)

data_directory = sys.argv[1]
sys.path.append('net/')

import net

# We delete previous results

from subprocess import call


_RESULTS_VALIDATION_LL = "/UCI_Datasets/" + data_directory + "/results/results_2_layers/validation_ll.txt"
_RESULTS_VALIDATION_RMSE = "/UCI_Datasets/" + data_directory + "/results/results_2_layers/validation_rmse.txt"
_RESULTS_VALIDATION_MC_RMSE = "/UCI_Datasets/" + data_directory + "/results/results_2_layers/validation_MC_rmse.txt"

_RESULTS_TEST_LL = "/UCI_Datasets/" + data_directory + "/results/results_2_layers/test_ll.txt"
_RESULTS_TEST_RMSE = "/UCI_Datasets/" + data_directory + "/results/results_2_layers/test_rmse.txt"
_RESULTS_TEST_MC_RMSE = "/UCI_Datasets/" + data_directory + "/results/results_2_layers/test_MC_rmse.txt"
_RESULTS_TEST_TIME = "/UCI_Datasets/" + data_directory + "/results/results_2_layers/time.txt"
_RESULTS_TEST_LOG = "/UCI_Datasets/" + data_directory + "/results/results_2_layers/log.txt"
_DROPOUT_RATES_FILE = "/UCI_Datasets/" + data_directory + "/results/dropout_rates.txt"
_TAU_VALUES_FILE = "/UCI_Datasets/" + data_directory + "/results/tau_values.txt"

_DATA_DIRECTORY_PATH = "/UCI_Datasets/" + data_directory + "/data/"
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
call(["rm", _RESULTS_TEST_LL])
call(["rm", _RESULTS_TEST_RMSE])
call(["rm", _RESULTS_TEST_MC_RMSE])
call(["rm", _RESULTS_TEST_TIME])
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

errors, MC_errors, lls, times = [], [], [], []
for i in range(n_splits):

    # We load the indexes of the training and test sets

    index_train = np.loadtxt(_get_index_train_test_path(i, train=True))
    index_test = np.loadtxt(_get_index_train_test_path(i, train=False))

    X_train = X[ [int(i) for i in index_train.tolist()] ]
    y_train = y[ [int(i) for i in index_train.tolist()] ]
    
    X_test = X[ [int(i) for i in index_test.tolist()] ]
    y_test = y[ [int(i) for i in index_test.tolist()] ]

    num_training_examples = int(0.8 * X_train.shape[0])
    X_validation = X_train[num_training_examples:, :]
    y_validation = y_train[num_training_examples:, :]
    X_train = X_train[0:(num_training_examples-1), :]
    y_train = y_train[0:(num_training_examples-1), :]

    # List of hyperparameters which we will try out using grid-search
    dropout_rates = np.loadtxt(_DROPOUT_RATES_FILE)
    tau_values = np.loadtxt(_TAU_VALUES_FILE)

    # We perform grid-search to select the best hyperparameters based on the highest log-likelihood value
    best_network = None
    best_ll = -float('inf')
    best_tau = 0
    best_dropout = 0
    for dropout_rate in dropout_rates:
        for tau in tau_values:
            print ('Cross validation step: Tau: ' + tau + ' Dropout rate: ' + dropout_rate)
            network = net.net(X_train, y_train,
                [ int(n_hidden), int(n_hidden) ], normalize = True, n_epochs = int(n_epochs*500), tau = tau,
                dropout = dropout_rate)
            running_time = network.running_time

            # We obtain the test RMSE and the test ll from the validation sets

            error, MC_error, ll = network.predict(X_validation, y_validation)
            if (ll > best_ll):
                best_ll = ll
                best_network = network
                best_tau = tau
                best_dropout = dropout_rate
            
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

            with open(_RESULTS_TEST_TIME, "a") as myfile:
                myfile.write(repr(running_time) + '\n')
            times += [running_time]

    # Storing test results
    error, MC_error, ll = best_network.predict(X_test, y_test)
    with open(_RESULTS_TEST_RMSE, "a") as myfile:
        myfile.write(repr(error) + '\n')

    with open(_RESULTS_TEST_MC_RMSE, "a") as myfile:
        myfile.write(repr(MC_error) + '\n')

    with open(_RESULTS_TEST_LL, "a") as myfile:
        myfile.write(repr(ll) + '\n')

    print ("Tests on split " + i " complete.")
    errors += [error]
    MC_errors += [MC_error]
    lls += [ll]

with open(_RESULTS_TEST_LOG, "a") as myfile:
    myfile.write('errors %f +- %f, median %f 25p %f 75p %f \n' % (np.mean(errors), np.std(errors), 
        np.percentile(errors, 50), np.percentile(errors, 25), np.percentile(errors, 75)))
    myfile.write('MC errors %f +- %f, median %f 25p %f 75p %f \n' % (np.mean(MC_errors), np.std(MC_errors), 
        np.percentile(MC_errors, 50), np.percentile(MC_errors, 25), np.percentile(MC_errors, 75)))
    myfile.write('lls %f +- %f, median %f 25p %f 75p %f \n' % (np.mean(lls), np.std(lls), 
        np.percentile(lls, 50), np.percentile(lls, 25), np.percentile(lls, 75)))
    myfile.write('times %f +- %f \n' % (np.mean(times), np.std(times)))
    myfile.write('tau %f \n' % (best_network.tau))
