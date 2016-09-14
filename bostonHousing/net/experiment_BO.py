# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by José Miguel Hernández-Lobato used for his 
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

# This is the Bayesian optimisation experiment used to find optimal model precision tau.
# This experiment uses Spearmint, obtained from here: https://github.com/JasperSnoek/spearmint/tree/master/spearmint/bin
# To run this experiment:
# ./cleanup path-to-this-folder
# ./spearmint path-to-this-folder/config.pb --driver=local --method=GPEIOptChooser --max-concurrent=1 --max-finished-jobs=30

import math

import numpy as np

folder = '../data'
import sys
sys.path.append('net_BO/')
import net


def run(tau): 
    # We load the data

    data = np.loadtxt(folder + '/data.txt')

    # We load the number of hidden units

    n_hidden = np.loadtxt(folder + '/n_hidden.txt').tolist()

    # We load the number of training epocs

    n_epochs = np.loadtxt(folder + '/n_epochs.txt').tolist()

    # We load the indexes for the features and for the target

    index_features = np.loadtxt(folder + '/index_features.txt')
    index_target = np.loadtxt(folder + '/index_target.txt')

    X = data[ : , index_features.tolist() ]
    y = data[ : , index_target.tolist() ]

    # We iterate over the training test splits

    # For BO we use 5 splits:
    n_splits = 5 #np.loadtxt('../data/n_splits.txt')

    errors, MC_errors, lls, times = [], [], [], []
    for i in range(n_splits):

        # We load the indexes of the training and test sets

        index = np.loadtxt(folder + '/index_train_{}.txt'.format(i))
        # For BO we use 20% of train as test:
        index_train = index[:int(len(index) * 0.8)]
        index_test = index[int(len(index) * 0.8):]

        X_train = X[ index_train.tolist(), ]
        y_train = y[ index_train.tolist() ]
        X_test = X[ index_test.tolist(), ]
        y_test = y[ index_test.tolist() ]

        # We construct the network

        # We iterate the method 

        network = net.net(X_train, y_train,
            [ n_hidden ], normalize = True, n_epochs = int(n_epochs * 1), X_test=X_test, y_test=y_test,
            tau = tau) 
        running_time = network.running_time

        # We obtain the test RMSE and the test ll

        error, MC_error, ll = network.predict(X_test, y_test)
        print i
        errors += [error]
        MC_errors += [MC_error]
        lls += [ll]
        times += [running_time]

    print np.mean(lls)
    return -1 * np.mean(lls)

# Write a function like this called 'main'
def main(job_id, params):
    # We fix the random seed
    np.random.seed(job_id)
    print 'Anything printed here will end up in the output directory for job #:', str(job_id)
    print params
    return run(params['tau'][0])

