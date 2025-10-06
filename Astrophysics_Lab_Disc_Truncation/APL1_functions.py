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
    df = pd.read_csv("C:\\Users\\casey\\Downloads\\single_star_disks_pop 3\\single_star_disks_pop" + str(file_path), delim_whitespace=True, header=None)
    # Print dataframe to check
    # print(df.head())
    return df

def make_pts(b, c, mu_indices):  
    """
    
    ok so you're not actually giving the mu value, but instead giving the indice that relates to each mu value.
    i.e. a mu of 0.1 would be indice [0]
    
    """
    p0 = np.array([b[mu_indices][0], c[mu_indices][0], 1e4], dtype=float)
    p1 = np.array([b[mu_indices][1], c[mu_indices][1], 1e5], dtype=float)
    p2 = np.array([b[mu_indices][2], c[mu_indices][2], 1e6], dtype=float)
    pts = np.vstack((p0, p1, p2))
    
    return pts


def Linear_interp(Re, b, c, mu_indices):
    if Re > make_pts(b, c, mu_indices)[0][2] and Re < make_pts(b, c, mu_indices)[1][2]: # if the given reynolds number is between the first two points, then b and c are can be interpolated
        t = (Re-make_pts(b, c, mu_indices)[0][2])/(make_pts(b, c, mu_indices)[1][2]-make_pts(b, c, mu_indices)[0][2])
        b_new = (1-t)*make_pts(b, c, mu_indices)[0][0]+t*make_pts(b, c, mu_indices)[1][0] 
        c_new = (1-t)*make_pts(b, c, mu_indices)[0][1]+t*make_pts(b, c, mu_indices)[1][1] 
        
        return (b_new, c_new)
    elif Re > make_pts(b, c, mu_indices)[1][2] and Re < make_pts(b, c, mu_indices)[2][2]:   # if the given reynolds number is between the second and third points, then b and c are can be interpolated
        t = (Re-make_pts(b, c, mu_indices)[1][2])/(make_pts(b, c, mu_indices)[2][2]-make_pts(b, c, mu_indices)[1][2])
        b_new = (1-t)*make_pts(b, c, mu_indices)[1][0]+t*make_pts(b, c, mu_indices)[2][0] 
        c_new = (1-t)*make_pts(b, c, mu_indices)[1][1]+t*make_pts(b, c, mu_indices)[2][1] 
        
        return (b_new, c_new)
    elif Re == make_pts(b, c, mu_indices)[0][2]:    # if the given reynolds number equals the first pt, then b and c are known
        b_new = make_pts(b, c, mu_indices)[0][0]
        c_new = make_pts(b, c, mu_indices)[0][1]
        return (b_new, c_new)
    elif Re == make_pts(b, c, mu_indices)[1][2]:    # if the given reynolds number equals the second pt, then b and c are known
        b_new = make_pts(b, c, mu_indices)[1][0]
        c_new = make_pts(b, c, mu_indices)[1][1]
        return (b_new, c_new)
    elif Re == make_pts(b, c, mu_indices)[2][2]:    # if the given reynolds number equals the third pt, then b and c are known
        b_new = make_pts(b, c, mu_indices)[2][0]
        c_new = make_pts(b, c, mu_indices)[2][1]
        return (b_new, c_new)
    else:                       # if conditions above fail, return error
        b_new = -27
        c_new = -27
        return (b_new, c_new)
    

def Quadratic_interp(Re, b, c, mu_indices):
    if Re > make_pts(b, c, mu_indices)[0][2] and Re < make_pts(b, c, mu_indices)[2][2]: # Interps new values for b and c based on a given reynolds number via the Lagrange polynomial interpolation procedure
        L0 = ((Re-make_pts(b, c, mu_indices)[1][2])*(Re-make_pts(b, c, mu_indices)[2][2]))/((make_pts(b, c, mu_indices)[0][2]-make_pts(b, c, mu_indices)[1][2])*(make_pts(b, c, mu_indices)[0][2]-make_pts(b, c, mu_indices)[2][2]))
        L1 = ((Re-make_pts(b, c, mu_indices)[0][2])*(Re-make_pts(b, c, mu_indices)[2][2]))/((make_pts(b, c, mu_indices)[1][2]-make_pts(b, c, mu_indices)[0][2])*(make_pts(b, c, mu_indices)[1][2]-make_pts(b, c, mu_indices)[2][2]))
        L2 = ((Re-make_pts(b, c, mu_indices)[0][2])*(Re-make_pts(b, c, mu_indices)[1][2]))/((make_pts(b, c, mu_indices)[2][2]-make_pts(b, c, mu_indices)[0][2])*(make_pts(b, c, mu_indices)[2][2]-make_pts(b, c, mu_indices)[1][2]))
        
        b_new = make_pts(b, c, mu_indices)[0][0]*L0 + make_pts(b, c, mu_indices)[1][0] *L1 + make_pts(b, c, mu_indices)[2][0]*L2
        c_new = make_pts(b, c, mu_indices)[0][1]*L0 + make_pts(b, c, mu_indices)[1][1] *L1 + make_pts(b, c, mu_indices)[2][1]*L2
        
        return (b_new, c_new)
    elif Re == make_pts(b, c, mu_indices)[0][2]:    # if the given reynolds number equals the first pt, then b and c are known
        b_new = make_pts(b, c, mu_indices)[0][0]
        c_new = make_pts(b, c, mu_indices)[0][1]
        return (b_new, c_new)
    elif Re == make_pts(b, c, mu_indices)[2][2]:    # if the given reynolds number equals the third pt, then b and c are known
        b_new = make_pts(b, c, mu_indices)[2][0]
        c_new = make_pts(b, c, mu_indices)[2][1]
        return (b_new, c_new)
    else:                       # if conditions above fail, return error
        b_new = -27
        c_new = -27
        return (b_new, c_new)
    


def R_egg(M1, M2, abin):
    q = [M1/M2, M2/M1]
    R_eggi = [] # Contains R eggleton for q1=M1/M2 and q2=M2/M1
    
    for i in [0,1]:
        R_egg = (0.49*q[i]**(2/3))*abin/(0.6*q[i]**(2/3)+np.log(1+q[i]**(1/3)))
        R_eggi.append(R_egg)
        
        
    return R_eggi

def Reynolds(alpha, aspect_ratio):
    Re = [] # Contains Reynolds numbers for given alpha and aspect ratio
    for ii in range(4):
        re = aspect_ratio[ii]**(-2) / alpha[ii]
        Re.append(re)
        
    return Re


def quad_interp_pts(Re, b, c, mu_indices):
    pts = Quadratic_interp(Re, b, c, mu_indices)
    
        
    b_new = pts[0]
    c_new = pts[1]
    return(b_new, c_new)

def linear_interp_pts(Re, b, c, mu_indices):
    pts = Linear_interp(Re, b, c, mu_indices)
    
        
    b_new = pts[0]
    c_new = pts[1]
    return(b_new, c_new)