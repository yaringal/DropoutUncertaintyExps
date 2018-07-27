This is the code used for the uncertainty experiments in the paper ["Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"](http://mlg.eng.cam.ac.uk/yarin/publications.html#Gal2015Dropout). This code is based on the code by José Miguel Hernández-Lobato used for his paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks". The datasets supplied here are taken from the UCI machine learning repository. Note the data splits used in these experiments (which are identical to the ones used in Hernández-Lobato's code). Because of the small size of the data, if you split the data yourself you will most likely get different and non-comparable results to the ones here.

To run an experiment:

```
THEANO_FLAGS='allow_gc=False,device=gpu,floatX=float32' python experiment.py --dir <UCI Dataset directory> --epochx <Epoch multiplier> --hidden <number of hidden layers>
```

The experiments were run with Theano 0.8.2 and Keras 2.2.0. The baseline experiment was to simply run the previous code (can be found in previous commits) with the new versions of Theano and Keras. The new experiments involved the following: 

1. Experiment 1 (No ensembling): Using 10x training epochs to train models on 20 randomly generated train-test splits of the data. The training set is further divided into an 80-20 train-validation split to find best hyperparameters, dropout rate and tau value through grid search. Finally, a network is trained on the whole training set using the best hyperparameters and is then tested on the test set. It's performance is reported.

2. Experiment 2 (Ensembling): Using 10x training epochs to train models on 20 randomly generated train-test splits of the data. The training set is further divided into an 80-20 train-validation split to find best hyperparameters, dropout rate and tau value through grid search. Finally, an ensemble of 10 networks is trained on the whole training set using the best hyperparameters and is then tested on the test set. The average performance of the ensemble is reported.

## 10x epochs results (compared to the original paper):
Dataset | Baseline RMSE | Experiment 1 RMSE | Experiment 2 RMSE | Baseline LL | Experiment 1 LL | Experiment 2 LL
--- | :---: | :---: | :---: | :---: | :---: | :---:
Boston Housing      | 2.83 ± 0.17 | 2.90 ± 0.18 | 2.82 ± 0.18 | -2.40 ± 0.04 | -2.40 ± 0.04 | -2.40 ± 0.05
Concrete Strength   | 4.93 ± 0.14 | 4.82 ± 0.16 | 4.82 ± 0.13 | -2.97 ± 0.02 | -2.93 ± 0.02 | -2.93 ± 0.02
Energy Efficiency   | 1.08 ± 0.03 | 0.54 ± 0.06 | 0.54 ± 0.01 | -1.72 ± 0.01 | -1.21 ± 0.01 | -1.21 ± 0.00
Kin8nm              | 0.09 ± 0.00 | 0.08 ± 0.00 | 0.08 ± 0.00 | 0.97 ± 0.00 | 1.14 ± 0.01 | 1.14 ± 0.00
Naval Propulsion    | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 3.91 ± 0.01 | 4.45 ± 0.00 | 4.45 ± 0.00
Power Plant         | 4.00 ± 0.04 | 4.01 ± 0.04 | 4.01 ± 0.04 | -2.79 ± 0.01 | -2.80 ± 0.01 | -2.80 ± 0.01
Protein Structure   | 4.27 ± 0.01 | 4.27 ± 0.02 | 4.26 ± 0.01 | -2.87 ± 0.00 | -2.87 ± 0.00 | -2.87 ± 0.00
Wine Quality Red    | 0.61 ± 0.01 | 0.62 ± 0.01 | 0.62 ± 0.01 | -0.92 ± 0.01 | -0.93 ± 0.01 | -0.93 ± 0.01
Yacht Hydrodynamics | 0.70 ± 0.05 | 0.67 ± 0.05 | 0.69 ± 0.04 | -1.38 ± 0.01 | -1.25 ± 0.01 | -1.26 ± 0.01

