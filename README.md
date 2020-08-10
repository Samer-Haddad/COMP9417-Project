# COMP9417-Project
**Airbnb New User Bookings (Kaggle Competition)**

**Link to competition:**
  https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/overview

**Link to all files:**
  https://drive.google.com/drive/folders/1CAMD8mVNjHKWsLxO5LIpswukDK_Vxtvy?usp=sharing

**This repository contains:**
  * main.py - script to clean data, train models, make predictions and output to predictions.csv (formatted by the competition's requirements).
  * predict.ipynb - jupyter notebook equivalent to main.py
  * predict_extended.ipynb - extensive parameters grid search for the models (extended and more detailed version of predict.ipynb).
  * /grid_search/ - contains graphs of parameters vs log loss (output from predict_extended.ipynb).
  * /datasets/datasets.zip/(train_users_2.csv, test_users.csv) -  raw data provided by the competition.
  * /datasets/datasets.zip/(df_impute.csv) - labelled and missing values imputations.
  * /datasets/datasets.zip/(X_train.csv, X_test.csv, y_train.csv, y_test.csv) - train-ready data split into train and validation sets.
  * /datasets/datasets.zip/(predictions.csv) - final predictions output using CatBoost model (formatted by the competition's requirements).

**In addition, COMP9417-Project-Files.zip on google drive contains pre-trained models:**
  * CatBoostClassifier
  * XGBClassifier
  * DecisionTreeClassifier
  * KNeighborsClassifier
  * RandomForestClassifier
  * SVM

**Note regarding predict_extended.ipynb:**<br/>
  GitHub does not display the tables side by side. Download and run in your IDE to view side by side tables in predict_extended.ipynb.

**Future suggestions for this problem that I wanted to pursue but didn't have enough time:**
  * Try to drop days coloumns and keep onyly day of week.
  * Calculate correlations.
  * Group ages into buckets to make age categorical.
  * I've extracted seasons from timeframes but dropped them before training due to high correlation with months. They would serve for visualisation purposes.
  * Try different dimensionality reduction techniques.
  * Treat -unknown- in gender coloumn as missing value.
