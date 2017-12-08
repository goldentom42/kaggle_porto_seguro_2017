##Porto Seguro's Safe Driver Challenge hosted by Kaggle

###Introduction
In late 2017, Kaggle and Porto Seguro, one of Brazil’s largest auto and homeowner 
insurance companies, organized a competition where kagglers were challenged to build
a model that predicts the probability that a driver will initiate an auto insurance 
claim in the next year. While Porto Seguro had used machine learning for the past 
20 years, they were looking to Kaggle’s machine learning community to explore new, 
more powerful methods. A more accurate prediction will allow them to further tailor
their prices, and hopefully make auto insurance coverage more accessible to more 
drivers.


###Models
In this competition I managed to finish at 33rd rank over 5000+ teams. My solution 
was based on the following set of models:

- [LightGBM](https://github.com/Microsoft/LightGBM)
- [XGBoost](https://github.com/dmlc/xgboost)
- [Regularized Greedy Forest](https://github.com/fukatani/rgf_python)
- [Feed-forward neural networks](https://keras.io/)
- [Field-Aware Factorization Machine](https://github.com/guestwalk/libffm), follow the link to the [original kernel by Scirpus]
- Follow The Regularized Leader Proximal, follow the link to the [original post and code by Scirpus]
- Stochastic Gradient Descent
- Ridge Classifier
- Logistic Regression

Most of these models used a subset of the dataset features whose selection was 
performed using my [py_ml_utils/feature_selector package](https://github.com/goldentom42/py_ml_utils)

Stacking was done using the linear stacker available [here](https://github.com/goldentom42/predictor_stacker)

###How to build the solution
Simply clone the repository and run ./build_solution.sh

###Dependencies
- Python 3.6
- Scikit-learn
- Pandas
- Keras with Theano backend
- LightGBM 0.6
- XGBoost 2.0.7
- LibFFM executables 
- https://github.com/goldentom42/py_ml_utils
- https://github.com/goldentom42/predictor_stacker


