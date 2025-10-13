import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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




def fit_for_b_and_c(b, c, mu_indices, plotting=False, xscale_log=True):
    known_mu = [0.1, 0.2, 0.3, 0.4, 0.5]

    # --- Build raw data (now we’ll want R itself later) ---
    points  = make_pts(b, c, mu_indices)        # [(b_i, c_i, Re_i), ...] length >= 3
    b_data  = np.array([points[i][0] for i in range(3)])
    c_data  = np.array([points[i][1] for i in range(3)])
    R_data  = np.array([points[i][2] for i in range(3)], dtype=float)  # Re
    x_data  = np.log(R_data)  # ln(Re)

    coeffs_b= np.polyfit(x_data, b_data, 2)
    print("Poly Fit Coefficients:", coeffs_b)
    # PBR = np.sqrt(np.diag(cov_b))

    coeffs_c= np.polyfit(x_data, c_data, 2)
    print("Poly Fit Coefficients:", coeffs_c)
    # PCR = np.sqrt(np.diag(cov_c))

    # Create polynomial function
    p_b = np.poly1d(coeffs_b)
    p_c = np.poly1d(coeffs_c)
    
    if plotting:
        # ---- Plot b vs Re (Re on x, b on y) ----
        plt.figure(figsize=(8, 5))
        plt.scatter(x_data, b_data, color='red', label='Data')
        x = np.linspace(np.min(x_data), np.max(x_data), 100)
        plt.plot(x, p_b(x),'b--', lw=2, label='Model from ln-fit')
        # plt.fill_between(R_fit, b_model - b_sigma, b_model + b_sigma, alpha=0.2, label='1σ band')
        if xscale_log:
            plt.xscale('log')
        plt.xlabel('Re')
        plt.ylabel('b')
        ttl = r'$\mathbf{b}$ vs $\mathrm{Re}$ (inverted from $\ln\mathrm{Re}=a\,e^{d\,b}$)' + f' for μ = {str(known_mu[mu_indices])}'
        plt.title(ttl)
        
        plt.figure(figsize=(8, 5))
        plt.scatter(x_data, c_data, color='red', label='Data')
        x = np.linspace(np.min(x_data), np.max(x_data), 100)
        plt.plot(x, p_c(x),'b--', lw=2, label='Model from ln-fit')
        # plt.fill_between(R_fit, b_model - b_sigma, b_model + b_sigma, alpha=0.2, label='1σ band')
        if xscale_log:
            plt.xscale('log')
        plt.xlabel('Re')
        plt.ylabel('b')
        ttl = r'$\mathbf{c}$ vs $\mathrm{Re}$ (inverted from $\ln\mathrm{Re}=a\,e^{d\,b}$)' + f' for μ = {str(known_mu[mu_indices])}'
        plt.title(ttl)
        

        plt.legend(); plt.tight_layout(); plt.show()


    return {
        "param_b": coeffs_b, "b fit": p_b,
        "param_c": coeffs_c, "c fit": p_c
    }
    
    
    
def find_and_plot_new_bc_points(b, c, mu_indices, R_targets, use_log_input=False, plotting=False, xscale_log=True):
    known_mu = [0.1, 0.2, 0.3, 0.4, 0.5]
    points  = make_pts(b, c, mu_indices)        # [(b_i, c_i, Re_i), ...] length >= 3
    b_data  = np.array([points[i][0] for i in range(3)])
    c_data  = np.array([points[i][1] for i in range(3)])
    R_data  = np.array([points[i][2] for i in range(3)], dtype=float)  # Re
    x_data  = np.log(R_data)  # ln(Re)

    # get polynomial fits from previous function
    fit_out = fit_for_b_and_c(b, c, mu_indices, plotting=False, xscale_log=xscale_log)
    p_b = fit_out["b fit"]
    p_c = fit_out["c fit"]

    # normalize targets
    if np.isscalar(R_targets):
        R_targets = [R_targets]
    R_targets = np.asarray(R_targets, dtype=float)
    x_targets = R_targets if use_log_input else np.log(R_targets)  # x = ln(Re)

    # compute new points from the polynomial fits
    new_b = p_b(x_targets)
    new_c = p_c(x_targets)

    # (Optional) estimate uncertainties — since polyfit doesn’t provide covariances directly,
    # you can leave them as zeros or compute them later via np.polyfit(cov=True)
    new_b_sigma = np.zeros_like(new_b)
    new_c_sigma = np.zeros_like(new_c)

    if plotting:
        min_val = np.min([np.min(x_data), np.min(x_targets)])
        max_val = np.max([np.max(x_data), np.max(x_targets)])
        plt.figure(figsize=(8,5))
        plt.scatter(x_targets, new_b, color='orange', label='New b* (polyfit)', zorder=3, alpha=.5)
        plt.scatter(x_data, b_data, color='blue', label='Known data', zorder=3, alpha=.5)
        plt.plot(np.linspace(min_val, max_val, 200),
                 p_b(np.linspace(min_val, max_val, 200)),
                 'b--', label='b fit')
        plt.xlabel(r'$\ln(\mathrm{Re})$' if not xscale_log else 'Re')
        plt.ylabel('b')
        plt.title(r'$\mathbf{b}$ vs $\ln(\mathrm{Re})$ using polynomial fit' + f' for μ = {str(known_mu[mu_indices])}')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

        plt.figure(figsize=(8,5))
        plt.scatter(x_targets, new_c, color='orange', label='New c* (polyfit)', zorder=3, alpha=.5)
        plt.scatter(x_data, c_data, color='blue', label='Known data', zorder=3, alpha=.5)
        plt.plot(np.linspace(min_val, max_val, 200),
                 p_c(np.linspace(min_val, max_val, 200)),
                 'b--', label='c fit')
        plt.xlabel(r'$\ln(\mathrm{Re})$' if not xscale_log else 'Re')
        plt.ylabel('c')
        plt.title(r'$\mathbf{c}$ vs $\ln(\mathrm{Re})$ using polynomial fit' + f' for μ = {str(known_mu[mu_indices])}')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    return {
        "new_b": new_b, "new_b_sigma": new_b_sigma,
        "new_c": new_c, "new_c_sigma": new_c_sigma,
        "p_b": p_b, "p_c": p_c
    }


    
def show_me_new_points(b, c, mu_indices, Re):
    
    known_mu = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    fits = find_and_plot_new_bc_points(b, c, mu_indices, Re, plotting=False)
    newb = []
    newc = []
    for ii in range(4):
        b_s = f"Interpolated values of b for μ = {str(known_mu[mu_indices])} are b = {fits['new_b'][ii]:.4f} ± {fits['new_b_sigma'][ii]:.4f} for Re = {Re[ii]}"
        c_s = f"Interpolated values of c for μ = {str(known_mu[mu_indices])} are c = {fits['new_c'][ii]:.4f} ± {fits['new_c_sigma'][ii]:.4f} for Re = {Re[ii]}"
        newb.append(b_s)
        newc.append(c_s)
    # print(newb)
    # print(newc)
    return newb, newc

def fit_mu(b, c, R_targets, mu_indices, plotting=False):
    
    known_mu = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

    # Normalize mu_indices to a list of ints (handles scalar, list, or array)
    mu_idx = np.atleast_1d(mu_indices)
    if not np.issubdtype(mu_idx.dtype, np.integer):
        mu_idx = mu_idx.astype(int)
    mu_idx = mu_idx.tolist()

    b_data, c_data = [], []
    for idx in mu_idx:
        out = find_and_plot_new_bc_points(b, c, idx, R_targets, plotting=False)
        b_data.append(out['new_b'][0])
        c_data.append(out['new_c'][0])

    mu_used = known_mu[mu_idx]  # μ values corresponding to the selected indices

    # Quadratic fits b(μ) and c(μ) using the selected μ points
    coeffs_b, cov_b = np.polyfit(mu_used, b_data, 2, cov=True)
    coeffs_c, cov_c = np.polyfit(mu_used, c_data, 2, cov=True)
    print("Poly fit coefficients for b(μ):", coeffs_b)
    print("Poly fit coefficients for c(μ):", coeffs_c)

    p_b = np.poly1d(coeffs_b)
    p_c = np.poly1d(coeffs_c)

    if plotting:
        x = np.linspace(np.min(mu_used), np.max(mu_used), 200)

        # Plot b vs μ
        plt.figure(figsize=(8, 5))
        plt.scatter(mu_used, b_data, color='red', label='Data')
        plt.plot(x, p_b(x), 'b--', lw=2, label='Quadratic fit')
        plt.xlabel(r'$\mu$')
        plt.ylabel(r'$b$')
        ttl_b = r'Fit of $b$ vs $\mu$ (from inverted model) | $\mu$ used: ' + ', '.join(map(str, np.round(mu_used, 3)))
        plt.title(ttl_b)
        plt.legend(); plt.tight_layout(); plt.show()

        # Plot c vs μ
        plt.figure(figsize=(8, 5))
        plt.scatter(mu_used, c_data, color='red', label='Data')
        plt.plot(x, p_c(x), 'b--', lw=2, label='Quadratic fit')
        plt.xlabel(r'$\mu$')
        plt.ylabel(r'$c$')  # fixed label
        ttl_c = r'Fit of $c$ vs $\mu$ (from inverted model) | $\mu$ used: ' + ', '.join(map(str, np.round(mu_used, 3)))
        plt.title(ttl_c)
        plt.legend(); plt.tight_layout(); plt.show()

    return {
        "coeffs_b": coeffs_b,
        "coeffs_c": coeffs_c,
        "poly_b": p_b,
        "poly_c": p_c,
        "mu_used": mu_used,
        "b_data": np.array(b_data),
        "c_data": np.array(c_data),
    }

  


def mu(M1, M2):
    return M2/(M1+M2)