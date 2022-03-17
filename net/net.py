# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his 
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

import warnings
warnings.filterwarnings("ignore")

import math
from scipy.special import logsumexp
import numpy as np

from keras.regularizers import l2
from keras import Input
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Softmax
from keras import Model
import tensorflow as tf

import time


class net:

    def __init__(self, X_train, y_train, n_hidden, n_epochs = 40,
        normalize = False, tau = 1.0, dropout = 0.05):

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
            @param tau          Tau value used for regularization
            @param dropout      Dropout rate for all the dropout layers in the
                                network.
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
        batch_size = 128
        lengthscale = 1e-2
        reg = lengthscale**2 * (1 - dropout) / (2. * N * tau)

        inputs = Input(shape=(X_train.shape[1],))
        inter = Dropout(dropout)(inputs, training=True)
        inter = Dense(n_hidden[0], activation='relu', kernel_regularizer=l2(reg))(inter)
        for i in range(len(n_hidden) - 1):
            inter = Dropout(dropout)(inter, training=True)
            inter = Dense(n_hidden[i+1], activation='relu', kernel_regularizer=l2(reg))(inter)
        inter = Dropout(dropout)(inter, training=True)
        outputs = Dense(2, kernel_regularizer=l2(reg))(inter)
        outputs = Softmax()(outputs)
        model = Model(inputs, outputs)

        model.compile(loss='binary_crossentropy', optimizer='adam')
        # print(model.summary())
        # We iterate the learning process
        start_time = time.time()
        model.fit(X_train, y_train_normalized, batch_size=batch_size, epochs=n_epochs, verbose=0)
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
        standard_pred_probs = model.predict(X_test, batch_size=500, verbose=1)
        standard_pred = tf.math.argmax(standard_pred_probs, axis=1).numpy()
        # standard_pred = standard_pred * self.std_y_train + self.mean_y_train
        # rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze())**2.)**0.5
        accuracy_standard_pred = np.mean((y_test.squeeze() == standard_pred.squeeze()))
        print(f'Standard Accuracy: {accuracy_standard_pred}')

        T = 100
        
        Yt_hat = np.array([model.predict(X_test, batch_size=500, verbose=0) for _ in range(T)])
        # Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train
        MC_pred = tf.math.argmax(np.mean(Yt_hat, axis=0), axis=1).numpy()
        # print(MC_pred.shape)
        mc_accuracy = np.mean((y_test.squeeze() == MC_pred.squeeze()))
        print(f'MC Accuracy: {mc_accuracy}')

        # We compute the test log-likelihood
        # ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat)**2., 0) - np.log(T) 
        #     - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        # test_ll = np.mean(ll)
        # ll = np.sum(y_test[])
        
        # double check this!
        y_test = y_test.astype(int)
        y_test_2d = np.hstack((y_test, 1-y_test))
        test_ll = np.mean(np.log(standard_pred_probs) * y_test_2d)

        # We are done!
        print(f'Test LL: {test_ll}')
        return accuracy_standard_pred, mc_accuracy, test_ll
