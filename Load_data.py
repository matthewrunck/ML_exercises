
# coding: utf-8

# '''
# This function loads and cleans data from Madelon and Gisette datasets
# '''

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


def load_data(train,train_labels,test,test_labels):
    #Load and clean  training data
    train_data = pd.read_csv(train, delimiter=' ',header=None)
    #remove empty columns
    train_data = train_data.dropna(axis='columns')
    
    
    train_mean = np.mean(train_data,axis=0)
    
    train_std= np.std(train_data,axis=0)
    
    #find columns with constant value for all obs
    col_to_drop = train_std==0
    
    
    #Normalise data
    X_train = (train_data-train_mean)/train_std
    
    #drop constant columns
    X_train = X_train.drop(X_train.columns[col_to_drop],axis=1)
    
    #add column of ones for intrcept term
    X_train = np.c_[np.ones(X_train.shape[0]),X_train]
   
    #Load training labels
    y_train = pd.read_csv(train_labels, delimiter=' ',header=None) 
    y_train = (y_train==1)

    
    #Load and clean testing data
    
    #Load test dataset
    test_data = pd.read_csv(test, delimiter=' ',header=None)
    test_data = test_data.dropna(axis='columns')
    
    #Apply normalization using mean and standard deviation of the training set
    X_test = (test_data-train_mean)/train_std
    
    #drop columns dropped from training set                
    X_test = X_test.drop(X_test.columns[col_to_drop],axis=1)
    
    #add column of ones for intercept
    X_test = np.c_[np.ones(X_test.shape[0]),X_test]
    
    #Load test_labels
    y_test = pd.read_csv(test_labels, delimiter=' ',header=None)
    y_test = y_test==1
    return X_train,y_train,X_test,y_test

