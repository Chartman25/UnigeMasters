import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from astropy.io import fits
import itertools
import corner
from typing import Iterable, Dict, List, Any
import sys
import os
code_dir = os.path.abspath(os.path.join(os.getcwd(), '..')) # Get directory above us
sys.path.append(code_dir) # Add directory above us to the path




def fits_data_loader(fresco_number, first_part=True):
    '''
    Loads in a csv from a given path. The csv must be in the Databases folder to work.
    '''
    # Load Fits
    fit = fits.open("C:\\Users\\casey\\Desktop\\2dspec\\fresco-only-n-v2_" +  str(fresco_number) + ".stack.fits", sep=r"\s+", header=None, engine="python")
    
    return fit