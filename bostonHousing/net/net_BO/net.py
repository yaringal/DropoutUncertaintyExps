# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his 
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

import warnings
warnings.filterwarnings("ignore")

import math
from scipy.misc import logsumexp
import numpy as np
from scipy import stats
from sklearn.preprocessing import scale

import pylab 
# %matplotlib notebook

import theano
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.constraints as constraints
from keras.callbacks import ModelTest, LearningRateScheduler

import time


class net:

    def __init__(self, X_train, y_train, n_hidden, n_epochs = 40,
        normalize = False, tau = 1., X_test = None, y_test = None):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[ self.std_X_train == 0 ] = 1
            self.mean_X_train = np.mean(X_train, 0)
        else:
            self.std_X_train = np.ones(X_train.shape[ 1 ])
            self.mean_X_train = np.zeros(X_train.shape[ 1 ])

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
            np.full(X_train.shape, self.std_X_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        y_train_normalized = np.array(y_train_normalized, ndmin = 2).T

        # We construct the network
        N = X_train.shape[0]
        dropout = 0.05
        batch_size = 32
        lengthscale = 1e-2
        reg = lambda: l2(lengthscale**2 * (1 - dropout) / (2. * N * tau))

        model = Sequential()
        model.add(Dropout(dropout))
        model.add(Dense(X_train.shape[1], n_hidden[0], W_regularizer = reg()))
        model.add(Activation('relu'))
        for i in xrange(len(n_hidden) - 1):
            model.add(Dropout(dropout))
            model.add(Dense(n_hidden[i], n_hidden[i+1], W_regularizer = reg()))
            model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(Dense(n_hidden[-1], y_train_normalized.shape[1], W_regularizer = reg()))

        # In Keras we have:
        # lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        # for SGD
        # optimiser = SGD(lr=lr, decay=SGD_decay, momentum=0.9, nesterov=False) 
        optimiser = 'adam'
        model.compile(loss='mean_squared_error', optimizer=optimiser)

        # We iterate the learning process
        start_time = time.time()
        X_test = np.array(X_test, ndmin = 2)
        y_test = np.array(y_test, ndmin = 2).T
        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)
        # modeltest = ModelTest(X_test, y_test, test_every_X_epochs=500, verbose=0, T=10, 
        #     loss='euclidean', mean_y_train=self.mean_y_train, std_y_train=self.std_y_train, tau=tau)
        # this doesn't seem to reduce error variance:
        # scheduler = LearningRateScheduler(lambda epoch: 0.005 / (epoch+1)**0.5)
        model.fit(X_train, y_train_normalized, batch_size=batch_size, nb_epoch=n_epochs, verbose=1) #, 
            # callbacks=[modeltest]) #scheduler, 
        self.model = model
        self.tau = tau
        self.running_time = time.time() - start_time

        # We are done!

    def predict(self, X_test, y_test):

        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data
            
    
            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.

        """

        X_test = np.array(X_test, ndmin = 2)
        y_test = np.array(y_test, ndmin = 2).T

        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        model = self.model
        standard_pred = model.predict(X_test, batch_size=500, verbose=1)
        standard_pred = standard_pred * self.std_y_train + self.mean_y_train
        rmse_standard_pred = np.mean((y_test - standard_pred)**2.)**0.5

        T = 10000

        Yt_hat = np.array([model.predict_stochastic(X_test, batch_size=500, verbose=0) for _ in xrange(T)])
        Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train
        MC_pred = np.mean(Yt_hat, 0)
        rmse = np.mean((y_test - MC_pred)**2.)**0.5

        # We compute the test log-likelihood
        ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat)**2., 0) - np.log(T) 
            - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        test_ll = np.mean(ll)
        
        print 'Standard rmse %f' % (rmse_standard_pred)
        print 'MC rmse %f' % (rmse)
        print 'test_ll %f' % (test_ll)

        # We are done!
        return rmse_standard_pred, rmse, test_ll
