# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:12:12 2018

@author: tjbanks

    phases = {"Cond"  : ["CS+/CS-", "eCS+/eCS-", "ICS+/ICS-"],
              "Cond_Shock"  : [""],
              "Ext"   : ["eCS+/eCS-", "ICS+/ICS-", "eeCS+/eeCS-", "IICS+/IICS-"],
              "Recall": ["eE/eU", "eE_eMinus"]}
    regions = ["vmPFC_ant", "vmPFC_post", "vmPFC_Iris", "Amygdala",
               "Hippo", "dACC", "Insula_L", "Insula_R"]
    
    Helpful resources:
    https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def get_data(filename, solution_num=3):
    if(solution_num not in range(3,6)):
        raise ValueError("Only three solutions (3,4,5)")
        
    pd_excel = pd.ExcelFile(filename)
    
    data = pd_excel.parse("fMRI")
    
    solutions = pd_excel.parse("Patient IDs",skiprows=13)
    solutions = solutions[solutions.columns[solution_num-1]]
    
    solutions = solutions.reset_index()
    data = data.reset_index()
    
    #Remove the columns that will have little effect/were not considered in this experiment
    remove_cols = ["Cond_CS", "Ext_ee", "Ext_ll","level_","index", "Patient ID"]
    for col in remove_cols:
        data = data[data.columns.drop(list(data.filter(regex=col)))]
        solutions = solutions[solutions.columns.drop(list(solutions.filter(regex=col)))]
        
    data.index += 1
    solutions.index += 1
    
    return data, solutions

def random_forest(train_features, train_labels, test_features, test_labels):
    rand_state = 42
    #Step 6: TRAIN
    #rf = RandomForestRegressor(n_estimators = 10000, random_state = 42)
    rf = RandomForestClassifier(n_estimators = 10000, random_state = rand_state)
    rf.fit(train_features, train_labels);
    
    
    #Step 7: TEST
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    print("Predictions: ")
    print(predictions)
    print("Test___data: ")
    print(test_labels)
    
    return accuracy_score(test_labels, predictions)
    
    #FOR REGRESSION
    # Calculate the absolute errors
    #errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    #print('Mean Absolute Error:', round(np.mean(errors), 2))
    
    #3 Score: 0.703703703704
    #4 Score: 0.518518518519
    #5 Score: 0.518518518519
    
def gradient_boosted_trees(train_features, train_labels, test_features, test_labels):
    
    return 0

def main():
    input_file = "Dataset_MGH_master_9_01_16.xlsx"
    num_solutions = 3
    
    #Step 1: Obtain the data
    try:
        (data, solutions) = get_data(input_file, solution_num=num_solutions)
    except ValueError as e:
        print(e)
        return
    
    #Step 2: Clean up
    #Do we have any missing data?
    #print(data.describe())
    #Can we either remove or replace with zeros?
    
    #MERGE (To keep solutions together)
    data_solutions = pd.concat([data,solutions],axis=1)
    
    #Drop all NAN Values
    data_solutions = data_solutions.dropna()
    #Drop all 0 solutions
    #data_solutions = data_solutions[data_solutions.columns[0] != 0]
        
    #SPLIT
    data = data_solutions.iloc[:, :-1]
    solutions = data_solutions.iloc[:,-1]
    
    #Step 3: Encoding
    #Look into one-hot-encoding, take readables, translate to nums in range
    #None necessary
    
    
    #Step 4: Feature and target separation, to arrays
    #targets already aquired
    
    labels = np.array(solutions)
    # Saving feature names for later use
    feature_list = list(data.columns)
    # Convert to numpy array
    features = np.array(data)
    
    
    #Step 5: Create Training and Test Sets
    test_size = .2
    rand_state = 42 #For reproducability
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size = test_size, random_state = rand_state)
    
    #print('Training Features Shape:', train_features.shape)
    #print('Training Labels Shape:', train_labels.shape)
    #print('Testing Features Shape:', test_features.shape)
    #print('Testing Labels Shape:', test_labels.shape)
    
    #SELECT ALGORITHM
    score = 0
    
    #RANDOM FOREST
    score = random_forest(train_features, train_labels, test_features, test_labels)
    
    #GRADIENT BOOSTED TREES
    score = gradient_boosted_trees(train_features, train_labels, test_features, test_labels)
    
    print("Accuracy Score: " + score)
    
main()