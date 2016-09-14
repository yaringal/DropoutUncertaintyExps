This is the code used for the uncertainty experiments in the paper ["Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"](http://mlg.eng.cam.ac.uk/yarin/publications.html#Gal2015Dropout). This code is based on the code by José Miguel Hernández-Lobato used for his paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks". The datasets supplied here are taken from the UCI machine learning repository. Note the data splits used in these experiments (which are identical to the ones used in Hernández-Lobato's code). Because of the small size of the data, if you split the data yourself you will most likely get different results to the ones here.

These experiments use Spearmint, obtained from here: [https://github.com/JasperSnoek/spearmint/tree/master/spearmint/bin](https://github.com/JasperSnoek/spearmint/tree/master/spearmint/bin).
To run an experiment:

```
./cleanup path-to-exp
THEANO_FLAGS='allow_gc=False,device=gpu,floatX=float32' ./spearmint path-to-exp/config.pb --driver=local --method=GPEIOptChooser --max-concurrent=1 --max-finished-jobs=30
```
