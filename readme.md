This is the code used for the uncertainty experiments in the paper ["Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" (2015)](http://www.cs.ox.ac.uk/people/yarin.gal/website/publications.html#Gal2015Dropout), with a few adaptions following recent (2018) feedback from the community (many thanks to @capybaralet for spotting some bugs, and @omegafragger for restructuring the code). This code is based on the code by José Miguel Hernández-Lobato used for his paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks". The datasets supplied here are taken from the UCI machine learning repository. Note the data splits used in these experiments (which are identical to the ones used in Hernández-Lobato's code). Because of the small size of the data, if you split the data yourself you will most likely get different and non-comparable results to the ones here.

**Update (2018)**
We replaced the Bayesian optimisation implementation (which was used to find hypers) with a grid-search over the hypers. This is following feedback from @capybaralet who spotted test-set contamination (some train-set points, used to tune hypers which were shared across all splits, were used as test-set points in later splits). The new implementation iterates over the 20 splits, and for each train-test split it creates a _new_ train-val split to tune hypers. These hypers are discarded between different train-test splits. 

Below we report the new results using grid-search (_new_, with code in this updated repo) vs. results obtained from a re-run of the original code used in the paper which used Bayesian optimisation (_paper_, code in [previous commit](https://github.com/yaringal/DropoutUncertaintyExps/tree/a6259f1db8f5d3e2d743f88ecbde425a07b12445)). Note that we report slightly different numbers for _paper_ than in the [previous commit](https://github.com/yaringal/DropoutUncertaintyExps/tree/a6259f1db8f5d3e2d743f88ecbde425a07b12445), due to differences in package versions and hardware from 3 years ago. Further note the improved results in _new_ on some datasets (mostly LL) due to proper grid-search (cases where BayesOpt failed). The other results agree with _paper_ within standard error. If you used the code from the [previous commits](https://github.com/yaringal/DropoutUncertaintyExps/tree/a6259f1db8f5d3e2d743f88ecbde425a07b12445) we advise you evaluate your method again following the stream-lined implementation here.

The experiments were run with Theano 0.8.2 and Keras 2.2.0. The baseline experiment (_paper_) was to simply run the previous "10x epochs one layer" code (can be found [here](https://github.com/yaringal/DropoutUncertaintyExps/tree/a6259f1db8f5d3e2d743f88ecbde425a07b12445)) with the new versions of Theano and Keras. 
The new code (_new_) uses 10x training epochs and one layer as well, and trains models on the same 20 randomly generated train-test splits of the data. Each training set is further divided into an 80-20 train-validation split to find best hyperparameters, dropout rate and tau value through grid search. Finally, a network is trained on the whole training set using the best hyperparameters and is then tested on the test set. 
To run an experiment:

```
THEANO_FLAGS='allow_gc=False,device=gpu,floatX=float32' python experiment.py --dir <UCI Dataset directory> --epochx <Epoch multiplier> --hidden <number of hidden layers>
```

A summary of the results is reported below (lower RMSE is better, higher test log likelihood (LL) is better; note the `±X` reported is _standard error_ and not standard deviation).

Dataset | BayesOpt RMSE (paper) | Grid Search RMSE (new) | BayesOpt LL (paper) | Grid Search LL (new)
--- | :---: | :---: | :---: | :---:
Boston Housing      | 2.83 ± 0.17 | 2.90 ± 0.18 | -2.40 ± 0.04 | -2.40 ± 0.04
Concrete Strength   | 4.93 ± 0.14 | 4.82 ± 0.16 | -2.97 ± 0.02 | -2.93 ± 0.02
Energy Efficiency   | 1.08 ± 0.03 | 0.54 ± 0.06 | -1.72 ± 0.01 | -1.21 ± 0.01
Kin8nm              | 0.09 ± 0.00 | 0.08 ± 0.00 | 0.97 ± 0.00 | 1.14 ± 0.01
Naval Propulsion    | 0.00 ± 0.00 | 0.00 ± 0.00 | 3.91 ± 0.01 | 4.45 ± 0.00
Power Plant         | 4.00 ± 0.04 | 4.01 ± 0.04 | -2.79 ± 0.01 | -2.80 ± 0.01
Protein Structure   | 4.27 ± 0.01 | 4.27 ± 0.02 | -2.87 ± 0.00 | -2.87 ± 0.00
Wine Quality Red    | 0.61 ± 0.01 | 0.62 ± 0.01 | -0.92 ± 0.01 | -0.93 ± 0.01
Yacht Hydrodynamics | 0.70 ± 0.05 | 0.67 ± 0.05 | -1.38 ± 0.01 | -1.25 ± 0.01

