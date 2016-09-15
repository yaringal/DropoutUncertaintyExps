# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his 
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

# This experiment uses the optimal model precision tau obtained from experiment_BO.py and 
# runs the model with a larger number of iterations.

import math

import numpy as np

import sys
sys.path.append('net/')
import net

# We delete previous results

from subprocess import call
call(["rm", "results_2_layers/test_ll.txt"])
call(["rm", "results_2_layers/test_rmse.txt"])
call(["rm", "results_2_layers/test_MC_rmse.txt"])
call(["rm", "results_2_layers/time.txt"])
call(["rm", "results_2_layers/log.txt"])

# We fix the random seed

np.random.seed(1)

# We load the data

data = np.loadtxt('../data/data.txt')

# We load the number of hidden units

n_hidden = np.loadtxt('../data/n_hidden.txt').tolist()

# We load the number of training epocs

n_epochs = np.loadtxt('../data/n_epochs.txt').tolist()

# We load the indexes for the features and for the target

index_features = np.loadtxt('../data/index_features.txt')
index_target = np.loadtxt('../data/index_target.txt')

X = data[ : , index_features.tolist() ]
y = data[ : , index_target.tolist() ]

# We iterate over the training test splits

n_splits = np.loadtxt('../data/n_splits.txt')

errors, MC_errors, lls, times = [], [], [], []
for i in range(n_splits):

    # We load the indexes of the training and test sets

    index_train = np.loadtxt("../data/index_train_{}.txt".format(i))
    index_test = np.loadtxt("../data/index_test_{}.txt".format(i))

    X_train = X[ index_train.tolist(), ]
    y_train = y[ index_train.tolist() ]
    X_test = X[ index_test.tolist(), ]
    y_test = y[ index_test.tolist() ]

    # We construct the network

    # We iterate the method 

    network = net.net(X_train, y_train,
        [ n_hidden, n_hidden ], normalize = True, n_epochs = int(n_epochs*500), X_test=X_test, y_test=y_test)
    running_time = network.running_time

    # We obtain the test RMSE and the test ll

    error, MC_error, ll = network.predict(X_test, y_test)

    with open("results_2_layers/test_rmse.txt", "a") as myfile:
        myfile.write(repr(error) + '\n')

    with open("results_2_layers/test_MC_rmse.txt", "a") as myfile:
        myfile.write(repr(MC_error) + '\n')

    with open("results_2_layers/test_ll.txt", "a") as myfile:
        myfile.write(repr(ll) + '\n')

    with open("results_2_layers/time.txt", "a") as myfile:
        myfile.write(repr(running_time) + '\n')

    print i
    errors += [error]
    MC_errors += [MC_error]
    lls += [ll]
    times += [running_time]

with open("results_2_layers/log.txt", "a") as myfile:
    myfile.write('errors %f +- %f, median %f 25p %f 75p %f \n' % (np.mean(errors), np.std(errors), 
        np.percentile(errors, 50), np.percentile(errors, 25), np.percentile(errors, 75)))
    myfile.write('MC errors %f +- %f, median %f 25p %f 75p %f \n' % (np.mean(MC_errors), np.std(MC_errors), 
        np.percentile(MC_errors, 50), np.percentile(MC_errors, 25), np.percentile(MC_errors, 75)))
    myfile.write('lls %f +- %f, median %f 25p %f 75p %f \n' % (np.mean(lls), np.std(lls), 
        np.percentile(lls, 50), np.percentile(lls, 25), np.percentile(lls, 75)))
    myfile.write('times %f +- %f \n' % (np.mean(times), np.std(times)))
    myfile.write('tau %f \n' % (network.tau))
