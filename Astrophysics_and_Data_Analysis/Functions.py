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
    df = pd.read_csv("C:\\Users\\casey\\UnigeMasters\\Astrophysics_and_Data_Analysis\Databases\\" + str(file_path),  usecols=['t','y', 'ey'])
    # Print dataframe to check
    # print(df.head())
    return df