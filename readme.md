This is the code used for the uncertainty experiments in the paper ["Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"](http://mlg.eng.cam.ac.uk/yarin/publications.html#Gal2015Dropout). This code is based on the code by José Miguel Hernández-Lobato used for his paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks". The datasets supplied here are taken from the UCI machine learning repository. Note the data splits used in these experiments (which are identical to the ones used in Hernández-Lobato's code). Because of the small size of the data, if you split the data yourself you will most likely get different and non-comparable results to the ones here.

To run an experiment:

```
THEANO_FLAGS='allow_gc=False,device=gpu,floatX=float32' python experiment.py
```

I updated the scripts to run with the latest version of Keras. I also added two new experiments: 

1. using 10x training epochs compared to the original paper (which gives a drastic improvement both in terms of RMSE and test log-likelihood). These are under `experiment_10x_epochs.py`.
2. using 2 layer networks (following the updated experiments in "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks"). These are under `experiment_2_layers.py`.


## 10x epochs results (compared to the original paper):

Dataset | Dropout RMSE (original) | Dropout RMSE (10x epochs) | Dropout Test LL (original) | Dropout Test LL (10x epochs)
--- | :---: | :---: | :---: | :---:
Boston Housing      | 2.97 ± 0.19 | 2.80 ± 0.19 | -2.46 ± 0.06 | -2.39 ± 0.05
Concrete Strength   | 5.23 ± 0.12 | 4.81 ± 0.14 | -3.04 ± 0.02 | -2.94 ± 0.02
Energy Efficiency   | 1.66 ± 0.04 | 1.09 ± 0.05 | -1.99 ± 0.02 | -1.72 ± 0.02
Kin8nm              | 0.10 ± 0.00 | 0.09 ± 0.00 | 0.95 ± 0.01 | 0.97 ± 0.01
Naval Propulsion    | 0.01 ± 0.00 | 0.00 ± 0.00 | 3.80 ± 0.01 | 3.92 ± 0.01
Power Plant         | 4.02 ± 0.04 | 4.00 ± 0.04 | -2.80 ± 0.01 | -2.79 ± 0.01
Protein Structure   | 4.36 ± 0.01 | 4.27 ± 0.01 | -2.89 ± 0.00 | -2.87 ± 0.00
Wine Quality Red    | 0.62 ± 0.01 | 0.61 ± 0.01 | -0.93 ± 0.01 | -0.92 ± 0.01
Yacht Hydrodynamics | 1.11 ± 0.09 | 0.72 ± 0.06 | -1.55 ± 0.03 | -1.38 ± 0.01


## 2 layers results (compared to the original paper):

Dataset | Dropout RMSE (original) | Dropout RMSE (2 layers) | Dropout Test LL (original) | Dropout Test LL (2 layers)
--- | :---: | :---: | :---: | :---:
Boston Housing      | 2.97 ± 0.19 | 2.80 ± 0.13 | -2.46 ± 0.06 | -2.34 ± 0.02
Concrete Strength   | 5.23 ± 0.12 | 4.50 ± 0.18 | -3.04 ± 0.02 | -2.82 ± 0.02
Energy Efficiency   | 1.66 ± 0.04 | 0.47 ± 0.01 | -1.99 ± 0.02 | -1.48 ± 0.00
Kin8nm              | 0.10 ± 0.00 | 0.08 ± 0.00 | 0.95 ± 0.01 | 1.10 ± 0.00
Naval Propulsion    | 0.01 ± 0.00 | 0.00 ± 0.00 | 3.80 ± 0.01 | 4.32 ± 0.00
Power Plant         | 4.02 ± 0.04 | 3.63 ± 0.04 | -2.80 ± 0.01 | -2.67 ± 0.01
Protein Structure   | 4.36 ± 0.01 | 3.62 ± 0.01 | -2.89 ± 0.00 | -2.70 ± 0.00
Wine Quality Red    | 0.62 ± 0.01 | 0.60 ± 0.01 | -0.93 ± 0.01 | -0.9 ± 0.01
Yacht Hydrodynamics | 1.11 ± 0.09 | 0.66 ± 0.06 | -1.55 ± 0.03 | -1.37 ± 0.02
