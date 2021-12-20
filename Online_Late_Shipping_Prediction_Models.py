#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 23:15:39 2021

@author: dysson
"""


# In[]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import tensorflow as tf

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from mixed_naive_bayes import MixedNB



# In[]:

# read data
data = pd.read_csv('ecommerce_shipping.csv')
data = data.drop('ID', axis=1)

# nulls?
print(data.isnull().sum())
# no nulls

# In[]:

'''
This function will separate the data into numerical and categorical type (object)
'''
def separate(data):
    num = data.dtypes != 'object'
    cat = data.dtypes == 'object'

    num = data[data.columns[num]]
    cat = data[data.columns[cat]]
    
    return num, cat

'''
This function is a modification of the script that I provided.
ixIt will match the categorical features in the training, validation, and testing sets simultaneously
Returns three cleaned data sets
'''
def cat_feat(train, valid, test):
    # Make sure categorical features in all sets match
    # Make sure the training, validation, and test features has same number of levels
    keep = [train.nunique()[i] == valid.nunique()[i] == test.nunique()[i] for i in range(train.shape[1])]
    train = train[train.columns[keep]]
    valid = valid[valid.columns[keep]]
    test = test[test.columns[keep]]

    # Make sure the levels are the same
    keep = []
    for i in range(train.shape[1]):
        keep.append(all(np.sort(train.iloc[:,i].unique()) == np.sort(valid.iloc[:,i].unique())) and all(np.sort(valid.iloc[:,i].unique()) == np.sort(test.iloc[:,i].unique())))
    train = train[train.columns[keep]]
    valid = valid[valid.columns[keep]]
    test = test[test.columns[keep]]
    return train, valid, test


# In[]:

# Split data into training and testing sets
X = data.drop('Reached.on.Time_Y.N', axis = 1)
y = data['Reached.on.Time_Y.N']

# convert obj cols to obj type
obj_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Customer_rating', 'Product_importance', 'Gender']
for i in obj_cols:
    X[i] = data[i].astype(object)

### First preprocess the data
# Separate numerical and categorical features
X_num, X_cat = separate(X)

# Convert categorical features to labels
X_cat = X_cat.apply(LabelEncoder().fit_transform)


# In[]:


# distribution of num data
for i in X_num:
    # raw data
    X_num[i].plot(kind='density')
    plt.xlabel(i)
    plt.title("Raw Data")
    plt.show()
    
    # log data
    log = np.log(1 + X_num[i])
    log.plot(kind='density', color='red')
    plt.xlabel(i)
    plt.title("Logged Data")
    plt.show()



# In[]:

# Log Transform some features to be more normally distributed
for i in X_num.columns:
    X_num['log_{}'.format(i)] = np.log(1 + X_num.loc[:,i])
    X_num = X_num.drop(i, axis=1)
    
# In[]:
    
# Label encode response
y = LabelEncoder().fit_transform(y)

# First combine the numerical and categorical features
fullX = pd.concat([X_num,X_cat], axis = 1)

# train/valid/test split
X_train_full, X_test, y_train_full, y_test = train_test_split(fullX, y, test_size = 0.2, random_state = 862)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size = 0.2, random_state = 862)


# In[]:

########################################################################################################
# Logistics Regression
########################################################################################################


# Logistics Regression
logreg = LogisticRegression()

# Set parameters
parameters = {'C' : [ 0.1, 1, 10, 50, 100]}

# fit model
logreg_cv = GridSearchCV(logreg, parameters, cv=10, n_jobs=-1)
logreg_cv.fit(X_train, y_train)

# What is your best combination?
print("LOGREG best params: ", logreg_cv.best_params_)
# LOGREG best params:  {'C': 50}

print("LOGREG accuracy = ", np.mean(logreg_cv.predict(X_valid) == y_valid))
# LOGREG accuracy =  0.6193181818181818


# In[]:

########################################################################################################
# SVM
########################################################################################################


# Linear

linear_svm = Pipeline([
                    ('svm_clf', LinearSVC(loss = "hinge", random_state = 42))
                    ])
linear_parameters = {'svm_clf__C' : [ 0.1, 1, 10, 100]}
linear_svm_clf = GridSearchCV(linear_svm, linear_parameters, cv=10, n_jobs=-1)
linear_svm_clf.fit(X_train, y_train)

# What is your best combination?
print("Linear SVM best params: ", linear_svm_clf.best_params_)
# Linear SVM best params:  {'svm_clf__C': 1}

print("Linear SVM accuracy = ", np.mean(linear_svm_clf.predict(X_valid) == y_valid))
# Linear SVM accuracy = 0.6488636363636363

########################################################################################################

# Polynomial

ploy_svm = Pipeline([
    ('svm_clf', SVC(kernel='poly'))
    ])
poly_parameters = {'svm_clf__C' : [ 0.1, 1, 10, 50],
              'svm_clf__degree': [1,2,3,4],
              'svm_clf__coef0': [0,1,2]}
poly_svm_clf = GridSearchCV(ploy_svm, poly_parameters, cv=10, n_jobs=-1)
poly_svm_clf.fit(X_train, y_train)

# What is your best combination?
print("Poly SVM best params: ", poly_svm_clf.best_params_)
# Poly SVM best params: {'svm_clf__C': 50, 'svm_clf__coef0': 2, 'svm_clf__degree': 3}

print("Poly SVM accuracy = ", np.mean(poly_svm_clf.predict(X_valid) == y_valid))
# Poly SVM accuracy =  0.6744318181818182

########################################################################################################

# RBF

rbf_svm = Pipeline([
    ('svm_clf', SVC(kernel='rbf'))
    ])

parameters = {'svm_clf__C' : [ 0.1, 1, 10, 100],
              'svm_clf__gamma': [0.1,0.5,1,2,3,4]}
rbf_svm_clf = GridSearchCV(rbf_svm, parameters, cv=10, n_jobs=-1)
rbf_svm_clf.fit(X_train, y_train)

# What is your best combination?
print("RBF SVM best params: ", rbf_svm_clf.best_params_)
# RBF SVM best params:  {'svm_clf__C': 1, 'svm_clf__gamma': 0.1}

print("RBF SVM accuracy = ", np.mean(rbf_svm_clf.predict(X_valid) == y_valid))
# RBF SVM accuracy =  0.6647727272727273

# In[]:

########################################################################################################
## Tree-Based Models
########################################################################################################

## DT

# set parameters
parameters = {'max_depth': [1,2,3,4,5,6,7,8,9,10],
              'criterion': ['gini', 'entropy']}

# Instantiate the model
dt_clf = DecisionTreeClassifier()

# fit model
dt_cv = GridSearchCV(dt_clf,parameters, cv = 10, n_jobs = -1)
dt_cv.fit(X_train, y_train)

# What is your best combination? Answer:
print("DT best params: ", dt_cv.best_params_)
# {'criterion': 'gini', 'max_depth': 5}

# Prediction
print("DT accuracy = ", np.mean(dt_cv.predict(X_valid) == y_valid))
# 0.6829545454545455

########################################################################################################

## RF

# set parameters
parameters = {'n_estimators' : [50, 100, 150, 200],
              'max_depth': [1,2,3,4,5,6,7,8,9,10],
              'criterion': ['gini', 'entropy']}

# Instantiate the model
rf_clf = RandomForestClassifier(random_state = 0)

# fit model
rf_cv = GridSearchCV(rf_clf,parameters, cv = 10, n_jobs = -1)
rf_cv.fit(X_train, y_train)

# What is your best combination? Answer:
print("RF best params: ", rf_cv.best_params_)
# {'criterion': 'entropy', 'max_depth': 5, 'n_estimators': 200}

# Prediction
print("RF accuracy = ", np.mean(rf_cv.predict(X_valid) == y_valid))
# 0.6835227272727272


########################################################################################################
# In[]:

## XGB

# set parameters
parameters = {'xgb_clf__n_estimators' : [50, 100, 150, 200],
              'xgb_clf__max_depth': [1,2,3,4,5,6,7,8,9,10],
              'xgb_clf__learning_rate': [0.1,0.5,1,2]}

# Instantiate the model
xgb_clf = XGBClassifier(n_jobs = -2, random_state = 862)

# fit model
xgb_cv = GridSearchCV(xgb_clf,parameters, cv = 10, n_jobs = -1)
xgb_cv.fit(X_train, y_train)

# What is your best combination? Answer:
print("XGB best params: ", xgb_cv.best_params_)
# {'xgb_clf__learning_rate': 0.1, 'xgb_clf__max_depth': 1, 'xgb_clf__n_estimators': 50}

# Prediction
print("XGB accuracy = ", np.mean(xgb_cv.predict(X_valid) == y_valid))
# 0.6630681818181818

# In[]:

########################################################################################################

## Light GBM

# set parameters
parameters = {'lgbm_clf__num_leaves' : [10,20,30,40,50],
              'lgbm_clf__learning_rate': [0.1,0.5,1,2],
              'lgbm_clf__n_estimators': [50, 100, 150, 200]}

# Instantiate the model
lgbm_clf = LGBMClassifier(n_jobs = -2, random_state = 862)

# fit model
lgbm_cv = GridSearchCV(lgbm_clf,parameters, cv = 10, n_jobs = -1)
lgbm_cv.fit(X_train, y_train)

# What is your best combination? Answer:
print("LGBM best params: ", lgbm_cv.best_params_)
# {'lgbm_clf__learning_rate': 0.1, 'lgbm_clf__n_estimators': 50, 'lgbm_clf__num_leaves': 10}

# Prediction
print("LGBM accuracy = ", np.mean(lgbm_cv.predict(X_valid) == y_valid))
# 0.6698863636363637


# In[]:

########################################################################################################
## Deep Learning Models
########################################################################################################

# NN
# First we will create a wrapper function
def build_model(n_hidden = 1, n_neurons = 30, activation_fcn = 'relu', optimizer = 'adam'):
    model = Sequential() # Instantiate the model
    input_dim = X_train.shape[1]
    options = {"input_dim": input_dim} # Set options 
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation = activation_fcn, **options)) # Here we are using the input options from before
        options = {} # Now we erase the input options so it won't be included in future layers
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizer,
              loss = 'binary_crossentropy', metrics = 'accuracy')
    return model


# Set up wrapper
keras_clf = tf.keras.wrappers.scikit_learn.KerasClassifier(build_model) 


# Set up the grid search
param = {'n_hidden': [1,2,3,4,5,6,7],
        'n_neurons':[10,20,30,100],
        'activation_fcn':['relu', 'sigmoid', 'linear'],
        'optimizer': ['SGD', 'adam']}

grd_cv = GridSearchCV(keras_clf, param, cv = 10, n_jobs = -1)


# Fit the model
grd_cv.fit(X_train, y_train, epochs = 100,
          validation_data = (X_valid, y_valid),
          callbacks = tf.keras.callbacks.EarlyStopping(patience=3))

# What is your best combination? Answer:
print("NN gridsearch best params: ", grd_cv.best_params_)
# NN gridsearch best params:  {'activation_fcn': 'sigmoid', 'n_hidden': 6, 'n_neurons': 10, 'optimizer': 'adam'}
# loss: 0.5285 - accuracy: 0.6488 - val_loss: 0.5353 - val_accuracy: 0.6506


print("NN gridsearch accuracy = ", np.mean(grd_cv.predict(X_test) == y_test)) 
# NN gridsearch accuracy = 0.46005289256198345

# In[]:

########################################################################################################
## Naive-Bayes Models
########################################################################################################

# GB
# Train GNB
X_train_GB, X_test_GB, y_train_GB, y_test_GB = train_test_split(X_num, y, test_size = 0.2, random_state = 862)

GNB = GaussianNB()
GNB.fit(X_train_GB, y_train_GB)
print("GB accuracy = ", np.mean(GNB.predict(X_test_GB) == y_test_GB))
# GB accuracy =  0.6431818181818182


########################################################################################################

# CB
# Train CNB
X_train_CB, X_test_CB, y_train_CB, y_test_CB = train_test_split(X_cat, y, test_size = 0.2, random_state = 862)

pipeline = Pipeline([('cnb', CategoricalNB())])
parameter = {'cnb__alpha':[0.1,0.5,1,2]}
CNB = GridSearchCV(pipeline, parameter, cv = 10)
CNB.fit(X_train_CB, y_train_CB)

# What is your best combination? Answer:
print("CB gridsearch best params: ", CNB.best_params_)
# CB gridsearch best params:  {'cnb__alpha': 0.1}

print("CB accuracy = ",np.mean(CNB.predict(X_test_CB) == y_test_CB))
# CB accuracy =  0.6036363636363636



########################################################################################################

# MNB
# X_train, X_test, y_train, y_test = train_test_split(fullX, y, test_size = 0.2, random_state = 862)
MNB = MixedNB(categorical_features = range(X_num.shape[1], fullX.shape[1]))
MNB.fit(X_train, y_train)
print("MNB accuracy = ", np.mean(MNB.predict(X_valid) == y_valid))
# MNB accuracy =  0.6568181818181819


# In[]:

########################################################################################################
## Ensemble Models
########################################################################################################


# define the base learners
logreg = LogisticRegression()
rf_clf = RandomForestClassifier(n_estimators=50, random_state=862)
rf_clf2 = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=862)
GB = GradientBoostingClassifier(random_state = 862)
DT = DecisionTreeClassifier(random_state = 862, max_depth=3)

# The tree-based models performed well so they will be the majority

########################################################################################################


## Stacked

# put base learners in a dictionary
models = {'logreg': logreg, 'rf_clf': rf_clf, 'rf_clf2': rf_clf2, 'GB': GB, 'DT': DT}

# define the blender
blender = LinearSVC()

# Train the weak learners
for name, model in models.items():
    model.fit(X_train, y_train)    
    
# Train the blender
# Get the prediction
predictions = pd.DataFrame() # Set up a dataframe to store the predictions
for name, model in models.items():
    predictions[name] = model.predict(X_valid)

# Get the blender
scaler_blend = StandardScaler() # Scale the predictions for SVR
predictions_scale = scaler_blend.fit_transform(predictions)
blender.fit(predictions_scale, y_valid)

# Perform evaluation
# First send the data through the weak learners
predictions = pd.DataFrame() # Set up a dataframe to store the predictions
for name, model in models.items():
    predictions[name] = model.predict(X_test)
    
# Prediction through the blender, and evaluate
predictions_scale = scaler_blend.transform(predictions)
print('Stacking:', np.mean(blender.predict(predictions_scale) == y_test))
# Final Stacking accuracy = 0.6772727272727272


########################################################################################################


## Voting

# Fit the voting regressor
vr = VotingClassifier(estimators = [('logreg', logreg), ('rf_clf', rf_clf), 
                                   ('GB', GB), ('DT', DT)], n_jobs = 2)

param = {'logreg__C' : [ 0.1, 1, 10, 50, 100],
         'rf_clf__n_estimators' : [50, 100, 150, 200],
         'rf_clf__max_depth': [1,2,3],
         'rf_clf__criterion': ['gini', 'entropy'],
         'GB__learning_rate':[0.01,0.1, 1, 10],
         'DT__max_depth': [1,2,3],
         'DT__criterion': ['gini', 'entropy']
         }

vr_cv = GridSearchCV(estimator = vr, param_grid = param, cv=2, n_jobs=-2)
vr_cv.fit(X_train, y_train)

print("VR accuracy = ", np.mean(vr_cv.predict(X_valid) == y_valid))
# VR accuracy =  0.6806818181818182

# In[]:

########################################################################################################
# FINAL MODEL
########################################################################################################


# The RF model gave us the best accuracy on the validation set so we will go with RF

# Instantiate the model
final_model = RandomForestClassifier(n_estimators=200, max_depth=5, criterion='entropy', random_state = 0)

# fit model on ALL training data
final_model.fit(X_train_full, y_train_full)

print("FINAL MODEL accuracy = ", np.mean(final_model.predict(X_test) == y_test))
# FINAL MODEL accuracy =  0.6731818181818182





