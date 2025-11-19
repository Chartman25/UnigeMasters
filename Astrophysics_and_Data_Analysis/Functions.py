import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
code_dir = os.path.abspath(os.path.join(os.getcwd(), '..')) # Get directory above us
sys.path.append(code_dir) # Add directory above us to the path




def data_loader(file_path):
    '''
    Loads in a csv from a given path. The csv must be in the Databases folder to work.
    '''
    # Load CSV
    df = pd.read_csv("C:\\Users\\casey\\UnigeMasters\\Astrophysics_and_Data_Analysis\\Databases\\" + str(file_path))
    # Print dataframe to check
    # print(df.head())
    return df


def min_maxing(features, range=0):
    '''
    Normalizes the data contained in features via min max scaling
    '''
    if range == 0:
        return (features - features.min()) / (features.max() - features.min())
    elif range == -1:
        return 2* (features - features.min()) / (features.max() - features.min()) -1

def z(features):
     '''
    Normalizes the data contained in features via standarization
    '''
     return (features - features.mean()) / features.std()
 
def scale_like_training(df, train_min, train_max):
    scaled = 2 * (df - train_min) / (train_max - train_min) - 1
    return scaled

def F1(CM, label=0):
    if label==0:
        precision = CM[0][0]/ (CM[0][0] + CM[1][0] +CM[2][0])
        recall = CM[0][0] / (CM[0][0] + CM[0][1] +CM[0][2])
    if label==1:
        precision = CM[1][1]/ (CM[1][1] + CM[0][1] +CM[2][1])
        recall = CM[1][1] / (CM[1][1] + CM[1][0] +CM[1][2])
    if label==2:
        precision = CM[2][2]/ (CM[2][2] + CM[0][2] +CM[1][2])
        recall = CM[2][2] / (CM[2][2] + CM[2][0] +CM[2][1])
    F1_score = 2 * (precision*recall) / (precision + recall) 
    return F1_score


def Confusion(known_labels, predicted_labels):
    CM_00=0
    CM_11=0
    CM_22=0
    CM_01=0
    CM_02=0
    CM_10=0
    CM_12=0
    CM_20=0
    CM_21=0
    for ii in range(len(predicted_labels)):
        if known_labels.iloc[ii]==0 and predicted_labels[ii]==0:
            CM_00 += 1 
        elif known_labels.iloc[ii]==1 and predicted_labels[ii]==1:
            CM_11 += 1 
        elif known_labels.iloc[ii]==2 and predicted_labels[ii]==2:
            CM_22 += 1 
        elif known_labels.iloc[ii]==0 and predicted_labels[ii]==1:
            CM_01 += 1 
        elif known_labels.iloc[ii]==0 and predicted_labels[ii]==2:
            CM_02 += 1 
        elif known_labels.iloc[ii]==1 and predicted_labels[ii]==0:
            CM_10 += 1 
        elif known_labels.iloc[ii]==1 and predicted_labels[ii]==2:
            CM_12 += 1 
        elif known_labels.iloc[ii]==2 and predicted_labels[ii]==0:
            CM_20 += 1 
        elif known_labels.iloc[ii]==2 and predicted_labels[ii]==1:
            CM_21 += 1 
        
        
        CM = [[CM_00,CM_01,CM_02],
              [CM_10,CM_11,CM_12],
              [CM_20,CM_21,CM_22]]
        
    return CM

def Accuracy(known_labels, predicted_labels):
    accuracy = np.sum(predicted_labels==known_labels)/len(known_labels)
    return accuracy