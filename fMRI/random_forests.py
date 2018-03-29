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
    Cross Validation: https://www.youtube.com/watch?v=TIgfjmp-4BA
    Keras Classifier: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

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

def dense_neural_network(train_features, train_labels, test_features, test_labels,num_labels):
    np.random.seed(7)
    rand_state = 42 
    
    #Put the thing back together
    train_features = np.append(train_features, test_features,axis=0)
    train_labels = np.append(train_labels, test_labels, axis=0)
    
    #Encode output
    encoder = LabelEncoder()
    encoder.fit(train_labels)
    
    encoded_train_labels = encoder.transform(train_labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_train_labels = np_utils.to_categorical(encoded_train_labels)

    #encoded_test_labels = encoder.transform(test_labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    #dummy_test_labels = np_utils.to_categorical(encoded_test_labels)
    
    def build_mod():
        # create model
        model = Sequential()
        model.add(Dense(30, input_dim=56, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(num_labels+1, activation='sigmoid'))
    	
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    # run model
    #model.fit(train_features, dummy_train_labels, epochs=150, batch_size=10)
    
    estimator = KerasClassifier(build_fn=build_mod, epochs=40, batch_size=5, verbose=1)
    kfold = KFold(n_splits=6, shuffle=True, random_state=rand_state)
    results = cross_val_score(estimator, train_features, dummy_train_labels, cv=kfold)

    return str.format("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
    #3 Fold Validation
    #3 Score: Accuracy Score: Baseline: 80.00% (4.50%)
    #4 Score: Accuracy Score: Baseline: 78.01% (2.12%)
    #5 Score: Accuracy Score: Baseline: 83.20% (1.18%)
    
    #6 Fold Validation
    #3 Score: Accuracy Score: Baseline: 77.26% (4.21%)
    #4 Score: Accuracy Score: Baseline: 77.97% (3.43%)
    #5 Score: Accuracy Score: Baseline: 81.07% (2.01%)
    
def main():
    input_file = "Dataset_MGH_master_9_01_16.xlsx"
    num_solutions = 5
    
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
    
    #### RANDOM FOREST
    #score = random_forest(train_features, train_labels, test_features, test_labels)
    
    #### GRADIENT BOOSTED TREES
    score = gradient_boosted_trees(train_features, train_labels, test_features, test_labels)
    
    #### DENSE NEURAL NETWORKS
    #score = dense_neural_network(train_features, train_labels, test_features, test_labels,num_solutions)
    
    print("Accuracy Score: " + str(score))
    
main()