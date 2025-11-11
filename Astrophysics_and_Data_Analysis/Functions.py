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
    df = pd.read_csv("C:\\Users\\casey\\UnigeMasters\\Astrophysics_and_Data_Analysis\Databases\\" + str(file_path))
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