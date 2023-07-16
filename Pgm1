import pandas as pd
import numpy as np

def find_s(con, tar):
    # Initialize the specific hypothesis to the first positive example
    for i, val in enumerate(tar):
        if val == 'yes':
            specific_h = con[i].copy()
            break
            
    # Loop through the positive examples and update the specific hypothesis
    for i, val in enumerate(con):
        if tar[i] == 'yes':
            for j in range(len(specific_h)):
                if val[j] != specific_h[j]:
                    specific_h[j] = '?'
                else:
                    pass
    
    return specific_h

def list_then_eliminate(con, tar):
    # Initialize the most general hypothesis
    general_h = ['?' for i in range(len(con[0]))]
    
    # Loop through the positive examples and eliminate any inconsistent attribute
    for i, val in enumerate(tar):
        if val == 'yes':
            for j in range(len(con[i])):
                if general_h[j] == '?':
                    general_h[j] = con[i][j]
                elif general_h[j] != con[i][j]:
                    general_h[j] = '?'
    
    return general_h


# Load the dataset from a CSV file
data = pd.read_csv('ENJOYSPORT.csv')
print(data) 

# Split the dataset into attributes and target variable
print("-----------")
concepts=np.array(data)[:,:-1]
print(concepts)
print("-----------")
targets=np.array(data)[:,-1]
print(targets)
print("-----------")
# Apply the FIND-S algorithm to find the specific hypothesis
h1 = find_s(concepts,targets)
print("Most specific hypothesis by FIND-S algorithm is:", h1)

# Apply the LIST THEN ELIMINATE algorithm to find the most specific hypothesis
h2 = list_then_eliminate(concepts,targets)
print("Final hypothesis space by LIST THEN ELIMINATE algorithm is:", h2)
