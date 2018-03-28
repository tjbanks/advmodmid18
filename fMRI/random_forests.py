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
    
    
"""

import pandas as pd

def get_data():
    return

def main():
    input_file = "Dataset_MGH_master_9_01_16.xlsx"
    sheets = ["fMRI","fMRI_transpose","Patient IDs"]
        
    data = pd.read_excel(input_file, sheet_name=sheets[0])
    solutions = pd.read_excel(input_file,sheet_name=sheets[2],skiprows=13)
    
    solutions3 = solutions.drop(columns=[solutions.columns[1], solutions.columns[3], solutions.columns[4], solutions.columns[5]])
    solutions4 = solutions.drop(columns=[solutions.columns[1], solutions.columns[2], solutions.columns[4], solutions.columns[5]])
    solutions5 = solutions.drop(columns=[solutions.columns[1], solutions.columns[2], solutions.columns[3], solutions.columns[5]])
    
    #Remove the columns that will have little effect/were not considered in this experiment
    
    solutions3 = solutions3.reset_index()
    data = data.reset_index()
    
    remove_cols = ["Cond_CS", "Ext_ee", "Ext_ll","level_","index", "Patient ID"]
    for col in remove_cols:
        data = data[data.columns.drop(list(data.filter(regex=col)))]
        solutions3 = solutions3[solutions3.columns.drop(list(solutions3.filter(regex=col)))]
        
    data.index += 1
    solutions3.index += 1
    
    
    print(data.head())
    print(solutions3.head())
    
    
main()