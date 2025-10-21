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




# def fit_for_b_and_c(b, c, mu_indices, plotting=False, xscale_log=True):
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
    
    
    
# def find_and_plot_new_bc_points(b, c, mu_indices, R_targets, use_log_input=False, plotting=False, xscale_log=True):
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


    
# def show_me_new_points(b, c, mu_indices, Re):
    
    known_mu = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    fits_b = find_and_plot_new_bc_points_re_and_mu(b, c, mu_indices, Re, Re_fit=True, plotting=False)
    newb = []
    newc = []
    for ii in range(4):
        b_s = f"Interpolated values of b for μ = {str(known_mu[mu_indices])} are b = {fits_b['new_b'][ii]:.4f} ± {fits_b['new_b_sigma'][ii]:.4f} for Re = {Re[ii]}"
        c_s = f"Interpolated values of c for μ = {str(known_mu[mu_indices])} are c = {fits_b['new_c'][ii]:.4f} ± {fits_b['new_c_sigma'][ii]:.4f} for Re = {Re[ii]}"
        newb.append(b_s)
        newc.append(c_s)
        
    # print(newb)
    # print(newc)
    return newb, newc

# def fit_mu(b, c, R_targets, mu_indices, plotting=False):
    
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
    coeffs_b, cov_b = np.polyfit(mu_used, b_data, 3, cov=True)
    coeffs_c, cov_c = np.polyfit(mu_used, c_data, 3, cov=True)
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




# def fit_b_c_mu_and_re(
#     b, c, mu_indices, R_targets=None, *,
#     Re_fit=True, Mu_fit=False,
#     plotting=False, xscale_log=True,
#     deg_mu=3,
#     re_idx=0   # <-- for Mu_fit: which of the 3 original points to use (0,1,2)
# ):
#     """
#     Re_fit=True:
#         Fit b,c vs ln(Re) for a SINGLE μ index using the same 3 original points.
#     Mu_fit=True (independent of Re-fit):
#         Fit b(μ) and c(μ) using the SAME original (b,c,Re) points.
#         We pick one of the 3 anchor points (re_idx in {0,1,2}) per μ so all μ's
#         are compared at the same anchor Re.

#     Returns:
#       Re_fit:
#         {
#           "mode": "Re_fit",
#           "param_b": coeffs_b, "b_fit": p_b,
#           "param_c": coeffs_c, "c_fit": p_c,
#           "R_data": R_data, "b_data": b_data, "c_data": c_data
#         }
#       Mu_fit:
#         {
#           "mode": "Mu_fit_anchor",
#           "re_idx": re_idx,
#           "anchor_Re": anchor_Re,       # Re at the chosen anchor
#           "coeffs_b": coeffs_b, "coeffs_c": coeffs_c,
#           "poly_b": p_b, "poly_c": p_c, # functions of μ
#           "mu_used": mu_used,
#           "b_data": np.array(b_data),   # b at anchor across μ
#           "c_data": np.array(c_data),   # c at anchor across μ
#         }
#     """
#     known_mu = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

#     # --- validate toggles ---
#     if (Re_fit and Mu_fit) or (not Re_fit and not Mu_fit):
#         raise ValueError("Set exactly one of Re_fit=True or Mu_fit=True.")

#     # ------------------------------------------------------------------
#     # MODE 1: Fit b,c as functions of Re for a single μ index (unchanged)
#     # ------------------------------------------------------------------
#     if Re_fit:
#         # Normalize to a single int μ index
#         if isinstance(mu_indices, (list, np.ndarray)):
#             if len(np.atleast_1d(mu_indices)) != 1:
#                 raise ValueError("Re_fit requires a single mu index (int).")
#             mu_idx = int(np.atleast_1d(mu_indices)[0])
#         else:
#             mu_idx = int(mu_indices)

#         # Same three original points
#         points  = make_pts(b, c, mu_idx)   # [(b_i, c_i, Re_i), ...]
#         b_data  = np.array([points[i][0] for i in range(3)], dtype=float)
#         c_data  = np.array([points[i][1] for i in range(3)], dtype=float)
#         R_data  = np.array([points[i][2] for i in range(3)], dtype=float)
#         x_data  = np.log(R_data)              # ln(Re)

#         # Quadratic in ln(Re)
#         coeffs_b = np.polyfit(x_data, b_data, 2)  # [A, B, C]
#         coeffs_c = np.polyfit(x_data, c_data, 2)
#         p_b = np.poly1d(coeffs_b)
#         p_c = np.poly1d(coeffs_c)

#         if plotting:
#             R_grid = (np.geomspace(np.min(R_data), np.max(R_data), 200)
#                       if xscale_log else
#                       np.linspace(np.min(R_data), np.max(R_data), 200))

#             # b vs Re
#             plt.figure(figsize=(8, 5))
#             plt.scatter(R_data, b_data, color='red', label='Data')
#             plt.plot(R_grid, p_b(np.log(R_grid)), 'b--', lw=2, label='Fit vs ln(Re)')
#             if xscale_log: plt.xscale('log')
#             plt.xlabel('Re'); plt.ylabel('b')
#             plt.title(r'$\mathbf{b}$ vs $\mathrm{Re}$ (fit in $\ln \mathrm{Re}$)'
#                       + f' | μ = {known_mu[mu_idx]}')
#             plt.legend(); plt.tight_layout(); plt.show()

#             # c vs Re
#             plt.figure(figsize=(8, 5))
#             plt.scatter(R_data, c_data, color='red', label='Data')
#             plt.plot(R_grid, p_c(np.log(R_grid)), 'b--', lw=2, label='Fit vs ln(Re)')
#             if xscale_log: plt.xscale('log')
#             plt.xlabel('Re'); plt.ylabel('c')
#             plt.title(r'$\mathbf{c}$ vs $\mathrm{Re}$ (fit in $\ln \mathrm{Re}$)'
#                       + f' | μ = {known_mu[mu_idx]}')
#             plt.legend(); plt.tight_layout(); plt.show()

#         return {
#             "mode": "Re_fit",
#             "param_b": coeffs_b, "b_fit": p_b,
#             "param_c": coeffs_c, "c_fit": p_c,
#             "R_data": R_data, "b_data": b_data, "c_data": c_data
#         }

#     # ------------------------------------------------------------------
#     # MODE 2: Fit b(μ), c(μ) using the SAME original (b,c,Re) points at a fixed anchor
#     # ------------------------------------------------------------------
#     if Mu_fit:
#         if re_idx not in (0, 1, 2):
#             raise ValueError("re_idx must be 0, 1, or 2 (one of the three original points).")

#         # Normalize μ indices to a list of ints
#         mu_idx_arr = np.atleast_1d(mu_indices)
#         if not np.issubdtype(mu_idx_arr.dtype, np.integer):
#             mu_idx_arr = mu_idx_arr.astype(int)
#         mu_idx_list = mu_idx_arr.tolist()
#         mu_used = known_mu[mu_idx_arr]

#         # Collect b,c at the chosen anchor across μ's
#         b_data, c_data = [], []
#         anchor_Re_vals = []
#         for idx in mu_idx_list:
#             pts = make_pts(b, c, idx)  # [(b_i, c_i, Re_i), ...]
#             b_val = float(pts[re_idx][0])
#             c_val = float(pts[re_idx][1])
#             Re_val = float(pts[re_idx][2])
#             b_data.append(b_val)
#             c_data.append(c_val)
#             anchor_Re_vals.append(Re_val)

#         # (Optional) sanity: all anchor Re should usually be identical across μ.
#         anchor_Re_vals = np.array(anchor_Re_vals, dtype=float)
#         anchor_Re = float(np.median(anchor_Re_vals))  # report representative anchor Re

#         # Fit polynomial in μ
#         coeffs_b = np.polyfit(mu_used, b_data, deg_mu)
#         coeffs_c = np.polyfit(mu_used, c_data, deg_mu)
#         p_b = np.poly1d(coeffs_b)
#         p_c = np.poly1d(coeffs_c)

#         if plotting:
#             x = np.linspace(np.min(mu_used), np.max(mu_used), 200)

#             # b vs μ
#             plt.figure(figsize=(8, 5))
#             plt.scatter(mu_used, b_data, color='red', label='Data')
#             plt.plot(x, p_b(x), 'b--', lw=2, label=f'Poly deg {deg_mu}')
#             plt.xlabel(r'$\mu$'); plt.ylabel(r'$b$')
#             plt.title(rf'Fit of $b$ vs $\mu$ at anchor $R_e \approx {anchor_Re:.3g}$ (idx {re_idx})')
#             plt.legend(); plt.tight_layout(); plt.show()

#             # c vs μ
#             plt.figure(figsize=(8, 5))
#             plt.scatter(mu_used, c_data, color='red', label='Data')
#             plt.plot(x, p_c(x), 'b--', lw=2, label=f'Poly deg {deg_mu}')
#             plt.xlabel(r'$\mu$'); plt.ylabel(r'$c$')
#             plt.title(rf'Fit of $c$ vs $\mu$ at anchor $R_e \approx {anchor_Re:.3g}$ (idx {re_idx})')
#             plt.legend(); plt.tight_layout(); plt.show()

#         return {
#             "mode": "Mu_fit_anchor",
#             "re_idx": re_idx,
#             "anchor_Re": anchor_Re,
#             "coeffs_b": coeffs_b,
#             "coeffs_c": coeffs_c,
#             "poly_b": p_b,
#             "poly_c": p_c,
#             "mu_used": mu_used,
#             "b_data": np.array(b_data, dtype=float),
#             "c_data": np.array(c_data, dtype=float),
#         }


def fit_b_c_mu_and_re(
    b, c, mu_indices, R_targets=None, *,
    Re_fit=True, Mu_fit=False,
    plotting=False, xscale_log=True,
    deg_mu=3,
    re_idx=0
):
    """
    Re_fit=True:
        Fit b,c vs ln(Re) for a SINGLE μ index using the same 3 original points.
    Mu_fit=True:
        Fit b(μ), c(μ) **using new points** obtained by first doing Re-fit at each μ
        and then sampling those fits at R_targets[re_idx] (back to the original approach).
        Requires R_targets and selects the anchor via re_idx.
    """
    known_mu = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

    # Validate toggles
    if (Re_fit and Mu_fit) or (not Re_fit and not Mu_fit):
        raise ValueError("Set exactly one of Re_fit=True or Mu_fit=True.")

    # -------------------------
    # Re-fit branch (unchanged)
    # -------------------------
    if Re_fit:
        # single μ index
        if isinstance(mu_indices, (list, np.ndarray)):
            mu_idx_arr = np.atleast_1d(mu_indices)
            if len(mu_idx_arr) != 1:
                raise ValueError("Re_fit requires a single mu index (int).")
            mu_idx = int(mu_idx_arr[0])
        else:
            mu_idx = int(mu_indices)

        # original 3 points
        points  = make_pts(b, c, mu_idx)  # [(b_i, c_i, Re_i), ...]
        b_data  = np.array([points[i][0] for i in range(3)], dtype=float)
        c_data  = np.array([points[i][1] for i in range(3)], dtype=float)
        R_data  = np.array([points[i][2] for i in range(3)], dtype=float)
        x_data  = np.log(R_data)

        # quadratic in ln(Re)
        coeffs_b = np.polyfit(x_data, b_data, 2)
        coeffs_c = np.polyfit(x_data, c_data, 2)
        p_b = np.poly1d(coeffs_b)
        p_c = np.poly1d(coeffs_c)

        if plotting:
            R_grid = (np.geomspace(R_data.min(), R_data.max(), 200)
                      if xscale_log else np.linspace(R_data.min(), R_data.max(), 200))

            # b vs Re
            plt.figure(figsize=(8, 5))
            plt.scatter(R_data, b_data, color='red', label='Data')
            plt.plot(R_grid, p_b(np.log(R_grid)), 'b--', lw=2, label='Fit vs ln(Re)')
            if xscale_log: plt.xscale('log')
            plt.xlabel('Re'); plt.ylabel('b')
            plt.title(r'$\mathbf{b}$ vs $\mathrm{Re}$ (fit in $\ln \mathrm{Re}$)'
                      + f' | μ = {known_mu[mu_idx]}')
            plt.legend(); plt.tight_layout(); plt.show()

            # c vs Re
            plt.figure(figsize=(8, 5))
            plt.scatter(R_data, c_data, color='red', label='Data')
            plt.plot(R_grid, p_c(np.log(R_grid)), 'b--', lw=2, label='Fit vs ln(Re)')
            if xscale_log: plt.xscale('log')
            plt.xlabel('Re'); plt.ylabel('c')
            plt.title(r'$\mathbf{c}$ vs $\mathrm{Re}$ (fit in $\ln \mathrm{Re}$)'
                      + f' | μ = {known_mu[mu_idx]}')
            plt.legend(); plt.tight_layout(); plt.show()

        return {
            "mode": "Re_fit",
            "param_b": coeffs_b, "b_fit": p_b,
            "param_c": coeffs_c, "c_fit": p_c,
            "R_data": R_data, "b_data": b_data, "c_data": c_data
        }

    # -----------------------------------------------------------
    # Mu-fit branch (reverted to use *new points* from Re-fitted)
    # -----------------------------------------------------------
    # Build b(μ) and c(μ) by first evaluating each μ's Re-fit at R_targets[re_idx]
    if R_targets is None:
        raise ValueError("Mu_fit requires R_targets.")
    R_targets = np.atleast_1d(R_targets).astype(float)
    if not (0 <= re_idx < len(R_targets)):
        raise IndexError("re_idx must select a valid entry in R_targets.")
    R_anchor = float(R_targets[re_idx])

    # normalize μ indices to a list
    mu_idx_arr = np.atleast_1d(mu_indices)
    if not np.issubdtype(mu_idx_arr.dtype, np.integer):
        mu_idx_arr = mu_idx_arr.astype(int)
    mu_idx_list = mu_idx_arr.tolist()
    mu_used = known_mu[mu_idx_arr]

    b_data, c_data = [], []
    for idx in mu_idx_list:
        # For each μ, get its Re-fit, then evaluate at the chosen R_anchor
        fit_re = find_and_plot_new_bc_points_re_and_mu(
            b, c,
            mu_indices=idx,
            R_targets=R_targets,
            Re_fit=True, Mu_fit=False,
            plotting=False
        )
        b_data.append(float(fit_re["new_b"][re_idx]))
        c_data.append(float(fit_re["new_c"][re_idx]))

    # Now fit polynomial in μ (degree deg_mu)
    coeffs_b = np.polyfit(mu_used, b_data, deg_mu, cov=False)
    coeffs_c = np.polyfit(mu_used, c_data, deg_mu, cov=False)
    p_b_mu = np.poly1d(coeffs_b)
    p_c_mu = np.poly1d(coeffs_c)

    if plotting:
        x = np.linspace(mu_used.min(), mu_used.max(), 200)
        # b vs μ
        plt.figure(figsize=(8,5))
        plt.scatter(mu_used, b_data, color='red', label='New b from Re-fit @ R', alpha=.7)
        plt.plot(x, p_b_mu(x), 'b--', lw=2, label=f'Poly deg {deg_mu}')
        plt.xlabel(r'$\mu$'); plt.ylabel('b')
        plt.title(rf'Fit of $b$ vs $\mu$ using Re-fit new points at $R_e={R_anchor:.3g}$')
        plt.legend(); plt.tight_layout(); plt.show()

        # c vs μ
        plt.figure(figsize=(8,5))
        plt.scatter(mu_used, c_data, color='red', label='New c from Re-fit @ R', alpha=.7)
        plt.plot(x, p_c_mu(x), 'b--', lw=2, label=f'Poly deg {deg_mu}')
        plt.xlabel(r'$\mu$'); plt.ylabel('c')
        plt.title(rf'Fit of $c$ vs $\mu$ using Re-fit new points at $R_e={R_anchor:.3g}$')
        plt.legend(); plt.tight_layout(); plt.show()

    return {
        "mode": "Mu_fit_from_Re_newpoints",
        "R_anchor": R_anchor,
        "mu_used": mu_used,
        "b_data": np.array(b_data, dtype=float),
        "c_data": np.array(c_data, dtype=float),
        "coeffs_b": coeffs_b, "coeffs_c": coeffs_c,
        "poly_b": p_b_mu, "poly_c": p_c_mu
    }



def find_and_plot_new_bc_points_re_and_mu(
    b, c,
    mu_indices,
    R_targets=None,
    use_log_input=False,
    plotting=False,
    xscale_log=True,
    *,
    Re_fit=True,
    Mu_fit=False,
    # Mu-fit specific:
    M1=None, M2=None,
    deg_mu=3,
    re_idx=0
):
    """
    Re_fit=True, Mu_fit=False  -> original behavior: fit vs ln(Re) for a single μ index and eval at R_targets.
    Re_fit=False, Mu_fit=True  -> μ-fit (independent): fit b(μ), c(μ) at a fixed anchor using original points,
                                 compute μ from (M1, M2), and evaluate there.

    Returns (Re_fit):
        {
          "mode": "Re_fit",
          "mu_index": mu_idx,
          "new_b": ..., "new_b_sigma": ...,
          "new_c": ..., "new_c_sigma": ...,
          "p_b": p_b (lnRe -> b), "p_c": p_c (lnRe -> c)
        }

    Returns (Mu_fit):
        {
          "mode": "Mu_fit",
          "mu_value": mu_value,
          "re_idx": re_idx,
          "anchor_Re": anchor_Re (approx),
          "new_b": array([b(mu_value)]), "new_c": array([c(mu_value)]),
          "new_b_sigma": 0s, "new_c_sigma": 0s,
          "p_b_mu": p_b_mu (μ -> b at anchor), "p_c_mu": p_c_mu (μ -> c at anchor)
        }
    """
    known_mu = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

    # --- validate toggles (same convention as before) ---
    if (Re_fit and Mu_fit) or (not Re_fit and not Mu_fit):
        raise ValueError("Set exactly one of Re_fit=True or Mu_fit=True.")

    # ------------------------------------------------------------------
    # Re-fit branch: unchanged core behavior (single μ index, fit vs ln Re)
    # ------------------------------------------------------------------
    if Re_fit:
        if mu_indices is None:
            raise ValueError("Re_fit requires a single mu index (int) in mu_indices.")
        # normalize to single int
        if isinstance(mu_indices, (list, np.ndarray)):
            mu_idx_arr = np.atleast_1d(mu_indices)
            if len(mu_idx_arr) != 1:
                raise ValueError("Re_fit requires a single mu index (int).")
            mu_idx = int(mu_idx_arr[0])
        else:
            mu_idx = int(mu_indices)

        # context points for plotting
        points  = make_pts(b, c, mu_idx)  # [(b_i, c_i, Re_i), ...] (>=3)
        b_data  = np.array([points[i][0] for i in range(3)], dtype=float)
        c_data  = np.array([points[i][1] for i in range(3)], dtype=float)
        R_data  = np.array([points[i][2] for i in range(3)], dtype=float)
        x_data  = np.log(R_data)

        # polynomial fits from your combined fitter (Re mode)
        fit_out = fit_b_c_mu_and_re(b, c, mu_idx, Re_fit=True, Mu_fit=False, plotting=False)
        p_b = fit_out["b_fit"]  # expects ln(Re)
        p_c = fit_out["c_fit"]

        # targets
        if R_targets is None:
            raise ValueError("Re_fit requires R_targets.")
        if np.isscalar(R_targets):
            R_targets = [R_targets]
        R_targets = np.asarray(R_targets, dtype=float)
        x_targets = R_targets if use_log_input else np.log(R_targets)

        # evaluate
        new_b = p_b(x_targets)
        new_c = p_c(x_targets)
        new_b_sigma = np.zeros_like(new_b)
        new_c_sigma = np.zeros_like(new_c)

        if plotting:
            xmin = np.min([np.min(x_data), np.min(x_targets)])
            xmax = np.max([np.max(x_data), np.max(x_targets)])
            xgrid = np.linspace(xmin, xmax, 200)

            # b plot
            plt.figure(figsize=(8,5))
            plt.scatter(x_targets, new_b, color='orange', label='New b* (polyfit)', zorder=3, alpha=.6)
            plt.scatter(x_data, b_data, color='blue', label='Known data', zorder=3, alpha=.6)
            plt.plot(xgrid, p_b(xgrid), 'b--', label='b fit')
            plt.xlabel(r'$\ln(\mathrm{Re})$' if not xscale_log else 'Re')
            plt.ylabel('b')
            plt.title(r'$\mathbf{b}$ vs $\ln(\mathrm{Re})$' + f' | μ = {known_mu[mu_idx]}')
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

            # c plot
            plt.figure(figsize=(8,5))
            plt.scatter(x_targets, new_c, color='orange', label='New c* (polyfit)', zorder=3, alpha=.6)
            plt.scatter(x_data, c_data, color='blue', label='Known data', zorder=3, alpha=.6)
            plt.plot(xgrid, p_c(xgrid), 'b--', label='c fit')
            plt.xlabel(r'$\ln(\mathrm{Re})$' if not xscale_log else 'Re')
            plt.ylabel('c')
            plt.title(r'$\mathbf{c}$ vs $\ln(\mathrm{Re})$' + f' | μ = {known_mu[mu_idx]}')
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

        return {
            "mode": "Re_fit",
            "mu_index": mu_idx,
            "new_b": new_b, "new_b_sigma": new_b_sigma,
            "new_c": new_c, "new_c_sigma": new_c_sigma,
            "p_b": p_b, "p_c": p_c
        }

    # ------------------------------------------------------------------
    # Mu-fit branch: independent of Re fit, uses μ(M1,M2) and anchor_idx
    # ------------------------------------------------------------------
    # normalize μ training indices (if None, use all)
    if mu_indices is None:
        mu_indices = [0, 1, 2, 3, 4]
    mu_idx_arr = np.atleast_1d(mu_indices)
    if not np.issubdtype(mu_idx_arr.dtype, np.integer):
        mu_idx_arr = mu_idx_arr.astype(int)

    if M1 is None or M2 is None:
        raise ValueError("Mu_fit requires M1 and M2 to compute μ.")
    mu_value = float(mu(M1, M2))

    # get μ-polys using same original anchor data (independent of Re fit)
    fit_out_mu = fit_b_c_mu_and_re(b, c, mu_idx_arr,Re_fit=False, Mu_fit=True,R_targets=R_targets,plotting=False,deg_mu=deg_mu,re_idx=re_idx)
    p_b_mu = fit_out_mu["poly_b"]   # μ -> b at anchor
    p_c_mu = fit_out_mu["poly_c"]   # μ -> c at anchor

    # evaluate at μ(M1,M2)
    new_b = np.array([p_b_mu(mu_value)], dtype=float)
    new_c = np.array([p_c_mu(mu_value)], dtype=float)
    new_b_sigma = np.zeros_like(new_b)
    new_c_sigma = np.zeros_like(new_c)

    if plotting:
        mu_used = fit_out_mu.get("mu_used", None)
        b_data_mu = fit_out_mu.get("b_data", None)
        c_data_mu = fit_out_mu.get("c_data", None)

        if (mu_used is not None) and (b_data_mu is not None):
            xx = np.linspace(np.min(mu_used), np.max(mu_used), 200)
            plt.figure(figsize=(8,5))
            plt.scatter(mu_used, b_data_mu, color='blue', label='Known b at anchor', alpha=.6)
            plt.plot(xx, p_b_mu(xx), 'b--', lw=2, label=f'Poly deg {deg_mu}')
            plt.scatter([mu_value], [new_b[0]], color='orange', zorder=3,
                        label=fr'New $b(\mu)$ at $\mu={mu_value:.3f}$')
            anch_Re = fit_out_mu.get("anchor_Re", None)
            title_b = r'$\mathbf{b}$ vs $\mu$'
            if anch_Re is not None:
                title_b += rf' (anchor $R_e \approx {anch_Re:.3g}$)'
            plt.xlabel(r'$\mu$'); plt.ylabel('b')
            plt.title(title_b)
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

        if (mu_used is not None) and (c_data_mu is not None):
            xx = np.linspace(np.min(mu_used), np.max(mu_used), 200)
            plt.figure(figsize=(8,5))
            plt.scatter(mu_used, c_data_mu, color='blue', label='Known c at anchor', alpha=.6)
            plt.plot(xx, p_c_mu(xx), 'b--', lw=2, label=f'Poly deg {deg_mu}')
            plt.scatter([mu_value], [new_c[0]], color='orange', zorder=3,
                        label=fr'New $c(\mu)$ at $\mu={mu_value:.3f}$')
            anch_Re = fit_out_mu.get("anchor_Re", None)
            title_c = r'$\mathbf{c}$ vs $\mu$'
            if anch_Re is not None:
                title_c += rf' (anchor $R_e \approx {anch_Re:.3g}$)'
            plt.xlabel(r'$\mu$'); plt.ylabel('c')
            plt.title(title_c)
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    return {
        "mode": "Mu_fit",
        "mu_value": mu_value,
        "re_idx": re_idx,
        "anchor_Re": fit_out_mu.get("anchor_Re", None),
        "new_b": new_b, "new_b_sigma": new_b_sigma,
        "new_c": new_c, "new_c_sigma": new_c_sigma,
        "p_b_mu": p_b_mu, "p_c_mu": p_c_mu
    }
  

def show_me_new_points(b, c, mu_indices, R_targets, M1, M2, *, deg_mu=3, re_idx=0,print_mu=True, print_Re=True):
    """
    Simple wrapper:
      - computes μ = M2/(M1+M2) and (optionally) prints b(μ), c(μ) at a chosen Re anchor
      - (optionally) prints b,c vs Re at the provided μ index (Re_fit)
      - controlled by print_mu / print_Re toggles
    """
    if not (print_mu or print_Re):
        raise ValueError("Set at least one of print_mu=True or print_Re=True.")

    known_mu = [0.1, 0.2, 0.3, 0.4, 0.5]
    newb, newc = [], []

    # μ-based single values (independent of Re-fit)
    if print_mu:
        fits_mu = find_and_plot_new_bc_points_re_and_mu(b, c,mu_indices=mu_indices,Re_fit=False, R_targets=R_targets, Mu_fit=True,M1=M1, M2=M2,deg_mu=deg_mu,re_idx=re_idx,plotting=False)
        mu_val = fits_mu["mu_value"]
        b_mu = fits_mu["new_b"][0]; b_mu_sig = fits_mu["new_b_sigma"][0]
        c_mu = fits_mu["new_c"][0]; c_mu_sig = fits_mu["new_c_sigma"][0]

        R = [1e4, 1e5, 1e6]  # existing convention with re_idx
        newb.append(f"Computed μ from masses: μ = {mu_val:.4f}  →  b(μ) = {b_mu:.4f} ± {b_mu_sig:.4f} for Re = {R[re_idx]:.3g})")
        newc.append(f"Computed μ from masses: μ = {mu_val:.4f}  →  c(μ) = {c_mu:.4f} ± {c_mu_sig:.4f} for Re = {R[re_idx]:.3g})")

    # Re-based interpolations at the chosen μ index
    if print_Re:
        fits_re = find_and_plot_new_bc_points_re_and_mu(b, c,mu_indices=mu_indices,R_targets=R_targets,Re_fit=True, Mu_fit=False,plotting=False)
        Re_arr = list(R_targets) if hasattr(R_targets, "__len__") and not isinstance(R_targets, (str, bytes)) else [R_targets]
        for ii in range(len(Re_arr)):
            b_s = (f"At μ_index={mu_indices} (μ≈{known_mu[mu_indices]}), Re={Re_arr[ii]}: "
                   f"b = {fits_re['new_b'][ii]:.4f} ± {fits_re['new_b_sigma'][ii]:.4f}")
            c_s = (f"At μ_index={mu_indices} (μ≈{known_mu[mu_indices]}), Re={Re_arr[ii]}: "
                   f"c = {fits_re['new_c'][ii]:.4f} ± {fits_re['new_c_sigma'][ii]:.4f}")
            newb.append(b_s)
            newc.append(c_s)

    return newb, newc


def truncation_radius(M1, M2, e, abin, b, c, R_targets, re_idx, plotting=False):
    """
    M1: Mass of star 1
    M2: Mass of star 2
    e: eccentricity of orbit
    abin: semi-major axis of the binary system
    b: values of b from table
    c: values of c from table
    R_targets: Reynolds number calculated from alpha, aspect ratio
    re_idx: [0,1,2] refers to test values of Re [1e4, 1e5, 1e6]
    """
    fit = find_and_plot_new_bc_points_re_and_mu(b, c, mu_indices=[0,1,2,3,4], R_targets=R_targets ,Re_fit=False, Mu_fit=True,M1=M1, M2=M2,re_idx=re_idx, deg_mu=3,plotting=False)
    
    
    R_eggi = R_egg(M1=M1, M2=M2, abin=abin)
    trunc = R_eggi[0]*(fit['new_b']*e**fit['new_c'] + 0.88*(M2/(M1+M2))**0.01)
    
    R = [1e4, 1e5, 1e6]

    print(f"the truncation radius for a given mu = {mu(M1,M2)} and given Reynolds number Re = {R[re_idx]} is {trunc}")
       
    if plotting:    
        # Create a blank canvas
        fig, ax = plt.subplots(figsize=(60, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        ax.axis('off')  # Hide axes

        # Write "CONGRATS" in a fun way
        ax.text(5, 1.5, f"CONGRATS! You found a truncation radius of {trunc}", 
                color='mediumspringgreen',
                fontsize=100,
                fontweight='bold',
                ha='center', va='center',
                family='sans-serif')

        # Add a confetti-like scatter effect
        np.random.seed(42)
        x_confetti = np.random.uniform(0, 10, 400)
        y_confetti = np.random.uniform(0, 3, 400)
        colors = np.random.choice(['gold', 'deeppink', 'deepskyblue', 'lime', 'orange'], 400)
        ax.scatter(x_confetti, y_confetti, s=40, c=colors, alpha=0.7)

        # Optional: make it celebratory
        ax.set_facecolor('black')

        plt.tight_layout()
        plt.show()                                                                          
    return trunc

def mu(M1, M2):
    return M2/(M1+M2)