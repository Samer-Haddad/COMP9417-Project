# COMP9417-Project
Airbnb New User Bookings (Kaggle Competition)

Link to competition:
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/overview

Pre-trained models, notebooks and supporting files are available on google drive:
https://drive.google.com/drive/folders/1CAMD8mVNjHKWsLxO5LIpswukDK_Vxtvy?usp=sharing

* main.py - script to clean data, train models, make predictions and output to predictions.csv (formatted by the competition's requirements).
* predict.ipynb - jupyter notebook equivalent to main.py
* predict_extended.ipynb - entensive parameters grid search for the models (extended and more detailed version of predict.ipynb).
* /grid_search/ - contains grpahs of parameters vs log loss (output from predict_extended.ipynb).
* /datasets/datasets.zip/(train_users_2.csv, test_users.csv) -  raw data provided by the competition.
* /datasets/datasets.zip/(df_impute.csv) - labelled and missing values imputations.
* /datasets/datasets.zip/(X_train.csv, X_test.csv, y_train.csv, y_test.csv) - train-ready data split to train and validation sets.
* /datasets/datasets.zip/(predictions.csv) - final predictions output using CatBoost model (formatted by the competition's requirements).
