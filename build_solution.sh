#!/usr/bin/env bash

# FTRL proximal
cd ftrl_proximal
python ftrl_proximal.py
# Keras shallow NN
cd ../keras_1layer/
python keras_dummies_csr.py
# Keras 2-layer NN
cd ../keras_2layers/
python keras_dummies_csr_deeper.py
# Light GBM
cd ../lgbm_full_features/
python lgbm_model.py
# Light GBM Random Forest
cd ../lightgbm_random_forest/
python lgbm_rf_model.py
# LibFFM
cd ../libffm
./run_lib_ffm.sh
# Logistic Regression
cd ../logistic_regression/
python logistic_regression.py
# Regularized Greedy Forest
cd ../regularized_greedy_forest/
python rgf_model.py
# Ridge Classifier
cd ../ridge_dummies/
python ridge_with_dummies.py
# Stochastic Gradient Descent
# Only use the 2-way combinations where all folds' scores improve
# This allows less overfit and stacking gets better
cd ../sgd_interactions/
python sgd_full_advance.py
# XGBoost
cd ../xgboost_interactions/
python xgboost_gini_model.py
# Now stack the whole thing
cd ../stacking/
python linear_stacking.py

