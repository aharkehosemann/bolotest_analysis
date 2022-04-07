import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize, minimize_scalar
# from scipy.stats import chi2

"""
Script for measuring individual layer contributions to the total thermal conductivity of a TES. 
Data from 'bolotest' measurement run in 2018. 
G_total and error is from TES power law fit at Tc, assuming dP/dT = G = n*k*Tc^(n-1)
Layers: U = SiN substrate, W = Nb wiring layers (<=2 layers per leg), I = SiN insulating layers (<=2 layers per leg)

aharkehosemann@gmail.com
2022/01

TODO: average quality plot with errors, then do 6 parameter minimization, error bars, alpha 0-2 and 0-1 versions of all these
"""

### User Switches
run_sim_allthree = False   # do minimizaion routines n_its times
run_testsim = False
random_initguess = False   # try simulation with randomized initial guesses
quality_plots = False   # plot chi-squared over parameter space
reduce_min = True   # reduce minimization to reconcile quality plots with chi_sq min
single_param = False
save_figs = False
show_plots = True

n_its = 1000   # number of iterations for MC simulation
num_guesses = 100   # number of randomized initial guesses to try
plot_dir = '/Users/angi/NIS/Analysis/bolotest/plots/layer_extraction_analysis/'

bolos = np.array(['bolo 1b', 'bolo 24', 'bolo 23', 'bolo 22', 'bolo 21', 'bolo 20', 'bolo 7', 'bolo 13'])
# p0 = np.array([0.89, 0.3, 1.3, .58, 1.4, 1.2])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]; Joel's results from chi-sq min sim
p0 = np.array([0.89, 0.3, 1.3, .58, 0.5, 0.5])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]; Joel's results from chi-sq min sim
ydata = np.array([11.7145073, 4.921841228, 8.077815536, 10.03001622, 16.63099617, 5.386790491, 15.2863792, 3.585251305])   # pW/K at Tc, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13*
sigma = np.array([0.100947739, 0.063601732, 0.078665632, 0.130040288, 0.142600818, 0.059261252, 0.123206779, 0.052084114])   # pW/K at Tc; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13*
# bounds1 = [(0, 0, 0, 0, 0, 0), (np.inf, np.inf, np.inf, 2, 2, 2)]; bounds2 = [(0, np.inf), (0, np.inf), (0, np.inf), (0, 2), (0, 2), (0, 2)]   # different fitting routines like different formats
bounds1 = [(0, 0, 0, 0, 0, 0), (np.inf, np.inf, np.inf, 1, 1, 1)]; bounds2 = [(0, np.inf), (0, np.inf), (0, np.inf), (0, 1), (0, 1), (0, 1)]   # different fitting routines like different formats
ydata_J = np.array([13.74, 4.921841228, 8.077815536, 10.03001622, 16.63099617, 5.386790491, 15.2863792, 3.585251305])   # pW/K at Tc, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13*
sigma_J = np.array([0.17, 0.063601732, 0.078665632, 0.130040288, 0.142600818, 0.059261252, 0.123206779, 0.052084114])   # pW/K at Tc; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13*
# ydata = ydata_J; sigma = sigma_J   # see if QPs match Joel's results
fits_J = np.array([.89, 0.30, 1.3277, 0.58, 1.42, 1.200]); error_J = np.array([0.02, 0.173, 0.0173, 0.548, 0.557, 0.0548])
boundsred = [(0, np.inf), (0, 1)]   # G, alpha
# boundsred = [(0, np.inf), (0, 2)]   # G, alpha
# boundsG = [(0, np.inf), (0, np.inf)]   # G, G

### Geometries
coeffs_all = np.array([[[4, 0, 0], [4, 4, 0], [4, 4, 0]],   # bolo 1b, consistent with Joel's numbers
        [[1, 3, 0], [1, 1, 0], [1, 1, 0]],   # bolo 24, consistent with Joel's numbers
        [[2, 2, 0], [2, 2, 0], [2, 2, 0]],   # bolo 23, consistent with Joel's numbers
        [[3, 1, 0], [3, 3, 0], [3, 3, 0]],   # bolo 22, consistent with Joel's numbers
        [[3, 1, 0], [1, 1, 0], [1, 3, 0]],   # bolo 21, consistent with Joel's numbers
        [[4, 0, 0], [3, 1, 1], [1, 0, 0]],   # bolo 20, consistent with Joel's numbers
        [[3, 1, 0], [3, 3, 0], [3, 1, 3]],   # bolo 7, consistent with Joel's numbers
        [[3, 1, 0], [1, 1, 0], [1, 0, 0]]])   # bolo 13, consistent with Joel's numbers

ratios_all = np.array([[[1, 0, 0], [160/400, 340/400, 0], [350/400, 1, 0]],   # bolo 1b, consistent with Joel's numbers
        [[1, 220/300, 0], [160/400, 340/400, 0], [350/400, 1, 0]],   # bolo 24, consistent with Joel's numbers
        [[1, 220/300, 0], [160/400, 340/400, 0], [350/400, 1, 0]],   # bolo 23, consistent with Joel's numbers
        [[1, 220/300, 0], [160/400, 340/400, 0], [350/400, 1, 0]],   # bolo 22, consistent with Joel's numbers
        [[280/300, 1, 0], [160/400, 285/400, 0], [350/400, (270+400)/400, 0]],   # bolo 21, consistent with Joel's numbers
        [[1, 0, 0], [(100+285)/400, 160/400, 285/400], [350/400, 0, 0]],   # bolo 20, consistent with Joel's numbers
        [[1, 280/300, 0], [160/400, 340/400, 0], [350/400, (270+400)/400, 1]],   # bolo 7, consistent with Joel's numbers
        [[220/300, 1, 0], [160/400, 285/400, 0], [350/400, 0, 0]]])   # bolo 13, consistent with Joel's numbers

xdata = np.array([coeffs_all, ratios_all])
data = [ydata, xdata, sigma]

### Supporting Functions
def G_layer(Gnorm, alpha, coeff, ratio):   # G of a single layer, to be summed over all three layers for one bolometer
    return Gnorm * np.nansum(coeff * ratio**(1+alpha))

def G_bolo(geom_terms, U, W, I, aU, aW, aI):   # G_total of a bolometer, sum of G_layer's
    coeffs, ratios = geom_terms
    return sum([G_layer(U, aU, coeffs[0], ratios[0]), G_layer(W, aW, coeffs[1], ratios[1]), G_layer(I, aI, coeffs[2], ratios[2])])

def chi_sq(ydata, ymodel, sigma):
    return np.nansum([((ydata[ii] - ymodel[ii])**2/ sigma[ii]**2) for ii in np.arange(len(ydata))])

def calc_chisq(params, data):   # wrapper for chi-squared min
    U, W, I, aU, aW, aI = params
    ydata, xdata, ysigma = data
    ymodel = [G_bolo([coeffs_all[ii], ratios_all[ii]], U, W, I, aU, aW, aI) for ii in np.arange(len(ydata))]
    return chi_sq(ydata, ymodel, ysigma)

def func_tomin(params, data):   # error is randomized every time
    GU, GW, GI, aU, aW, aI = params
    ydata, xdata, ysigma = data
    """
    ### with random errors added in
    bolo1b = (4*GU + GW*(4*(160/400)**(1 + aW) + 4*(340/400)**(1 + aW)) + GI*(4*(350/400)**(1 + aI) + 4) - ydata[0] + np.random.normal(scale=ysigma[0]))**2/ysigma[0]**2 
    bolo24 = (GU*(1 + 3*(220/300)**(1 + aU)) + GW*((160/400)**(1 + aW) + (340/400)**(1 + aW)) + GI*((350/400)**(1 + aI) + 1) - ydata[1] + np.random.normal(scale=ysigma[1]))**2/ysigma[1]**2 
    bolo23 = (GU*(2 + 2*(220/300)**(1 + aU)) + GW*(2*(160/400)**(1 + aW) + 2*(340/400)**(1 + aW)) + GI*(2*(350/400)**(1 + aI) + 2) - ydata[2] + np.random.normal(scale=ysigma[2]))**2/ysigma[2]**2
    bolo22 = (GU*(3 + (220/300)**(1 + aU)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)) + GI*(3*(350/400)**(1 + aI) + 3) - ydata[3] + np.random.normal(scale=ysigma[3]))**2/ysigma[3]**2 
    bolo21 = (GU*(1 + 3*(280/300)**(1 + aU)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)) + GI*(3*(670/400)**(1 + aI) + (350/400)**(1 + aI)) - ydata[4] + np.random.normal(scale=ysigma[4]))**2/ysigma[4]**2  
    bolo20 = (4*GU + GW*((160/400)**(1 + aW) + 3*(385/400)**(1 + aW) + (285/400)**(1 + aW)) + GI*(350/400)**(1 + aI) - ydata[5] + np.random.normal(scale=ysigma[5]))**2/ysigma[5]**2  
    bolo13 = (GU*(3 + (280/300)**(1 + aU)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)) + GI*(3*(350/400)**(1 + aI) + (670/400)**(1 + aI) + 3) - ydata[6] + np.random.normal(scale=ysigma[6]))**2/ysigma[6]**2 
    bolo7 = (GU*(1 + 3*(220/300)**(1 + aU)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)) + GI*(350/400)**(1 + aI) - ydata[7] + np.random.normal(scale=ysigma[7]))**2/ysigma[7]**2
    """ 
    ### without random errors 
    bolo1b = (4*GU + GW*(4*(160/400)**(1 + aW) + 4*(340/400)**(1 + aW)) + GI*(4*(350/400)**(1 + aI) + 4) - ydata[0])**2/ysigma[0]**2 
    bolo24 = (GU*(1 + 3*(220/300)**(1 + aU)) + GW*((160/400)**(1 + aW) + (340/400)**(1 + aW)) + GI*((350/400)**(1 + aI) + 1) - ydata[1])**2/ysigma[1]**2 
    bolo23 = (GU*(2 + 2*(220/300)**(1 + aU)) + GW*(2*(160/400)**(1 + aW) + 2*(340/400)**(1 + aW)) + GI*(2*(350/400)**(1 + aI) + 2) - ydata[2])**2/ysigma[2]**2
    bolo22 = (GU*(3 + (220/300)**(1 + aU)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)) + GI*(3*(350/400)**(1 + aI) + 3) - ydata[3])**2/ysigma[3]**2 
    bolo21 = (GU*(1 + 3*(280/300)**(1 + aU)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)) + GI*(3*(670/400)**(1 + aI) + (350/400)**(1 + aI)) - ydata[4])**2/ysigma[4]**2  
    bolo20 = (4*GU + GW*((160/400)**(1 + aW) + 3*(385/400)**(1 + aW) + (285/400)**(1 + aW)) + GI*(350/400)**(1 + aI) - ydata[5])**2/ysigma[5]**2  
    bolo13 = (GU*(3 + (280/300)**(1 + aU)) + GW*(3*(160/400)**(1 + aW) + 3*(340/400)**(1 + aW)) + GI*(3*(350/400)**(1 + aI) + (670/400)**(1 + aI) + 3) - ydata[6])**2/ysigma[6]**2 
    bolo7 = (GU*(1 + 3*(220/300)**(1 + aU)) + GW*((160/400)**(1 + aW) + (285/400)**(1 + aW)) + GI*(350/400)**(1 + aI) - ydata[7])**2/ysigma[7]**2
    
    return np.sum([bolo1b, bolo24, bolo23, bolo22, bolo21, bolo20, bolo13, bolo7])  

def calc_chisq_grid(params, data):   # chi-squared parameter space
    ydata, xdata, ysigma = data
    chisq_grid = np.full((len(params), len(params)), np.nan)
    for rr, row in enumerate(params): 
        for cc, col in enumerate(row):
            U, W, I, aU, aW, aI = col
            ymodel = [G_bolo([coeffs_all[ii], ratios_all[ii]], U, W, I, aU, aW, aI) for ii in np.arange(len(ydata))]
            chisq_grid[rr, cc] = chi_sq(ydata, ymodel, ysigma)
    return chisq_grid

def calc_func_grid(params, data):   # chi-squared parameter space
    # ydata, xdata, ysigma = data
    func_grid = np.full((len(params), len(params)), np.nan)
    for rr, row in enumerate(params): 
        for cc, col in enumerate(row):
            # U, W, I, aU, aW, aI = col
            params_rc = col            
            # params = U, W, I, aU, aW, aI
            # ymodel = [G_bolo([coeffs_all[ii], ratios_all[ii]], U, W, I, aU, aW, aI) for ii in np.arange(len(ydata))]
            func_grid[rr, cc] = func_tomin(params_rc, data)
    return func_grid

def run_sim(num_its, p0, ydata, xdata, sigma, bounds1, bounds2, show_yplots=False, save_figs=False):   ### MC sim of LS & CS minimization

    print('Running minimization simulation over three routines.')

    ### Least Squares Min
    pfit_LS, pcov_LS = curve_fit(G_bolo, xdata, ydata, p0=p0, sigma=sigma, absolute_sigma=True, bounds=bounds1)   # non-linear least squares fit
    U_LS, W_LS, I_LS, aU_LS, aW_LS, aI_LS = pfit_LS   # best fit parameters
    perr_LS = np.sqrt(np.diag(pcov_LS)); Uerr_LS = perr_LS[0]; Werr_LS = perr_LS[1]; Ierr_LS = perr_LS[2]; aUerr_LS = perr_LS[3]; aWerr_LS = perr_LS[4]; aIerr_LS = perr_LS[5]   # error of fit

    ### Chi-Squared Min
    chi_result = minimize(calc_chisq, p0, args=[ydata, xdata, sigma], bounds=bounds2)   # fit is unsuccessful and results are nonsense if bounds aren't specified 
    # chi_result = minimize(calc_chisq, p0, args=[ydata, xdata, sigma])  
    cbounds_met = np.array([bounds2[ii][0]<=chi_result['x'][ii]<=bounds2[ii][1] for ii in np.arange(len(chi_result['x']))]).all()
    if chi_result['success']:       
        U_CS, W_CS, I_CS, aU_CS, aW_CS, aI_CS = chi_result['x']
    else:
        print('Chi-Squared Min was unsuccessful.')

    ### Hand-Written Function Minimization
    func_result = minimize(func_tomin, p0, args=[ydata, xdata, sigma], bounds=bounds2) 
    if func_result['success']:   
        fresult_temp = func_result['x']
        fbounds_met = np.array([bounds2[ii][0]<=func_result['x'][ii]<=bounds2[ii][1] for ii in np.arange(len(func_result['x']))]).all()
        if ~fbounds_met:   # check if fit parameters are not within bounds
            print('Some or all fit parameters returned were not within the prescribed bounds. \n Changing these to NaNs. \n')
            fresult_temp[~fbounds_met] = np.nan
        U_func, W_func, I_func, aU_func, aW_func, aI_func = fresult_temp
    else:
        print('Single Function Min was unsuccessful.')

    pfits_LS = np.empty((n_its, len(p0))); pfits_CS = np.empty((n_its, len(p0))); pfits_func = np.empty((n_its, len(p0)))
    y_its = np.empty((n_its, len(ydata)))
    for ii in np.arange(num_its):   # run simulation
        # least squares
        y_it = np.random.normal(ydata, sigma)   # pull G's from normal distribution characterized by fit error
        # y_it = np.random.uniform(low=0, high=20, size=len(ydata))   # for testing
        y_its[ii] = y_it
        pfit_LS, pcov_LS = curve_fit(G_bolo, xdata, y_it, p0=p0, sigma=sigma, absolute_sigma=True, bounds=bounds1)
        pfits_LS[ii] = pfit_LS

        # chi-squared
        chi_result = minimize(calc_chisq, p0, args=[y_it, xdata, sigma], bounds=bounds2) 
        if chi_result['success']:   
            pfits_CS[ii] = chi_result['x']
            cbounds_met = np.array([bounds2[ii][0]<=chi_result['x'][ii]<=bounds2[ii][1] for ii in np.arange(len(chi_result['x']))]).all()
            if ~cbounds_met:   # check if fit parameters are not within bounds
                print('Some or all Chi-Squared fit parameters returned were not within the prescribed bounds. \n Setting to NaNs. \n')
                pfits_CS[ii] = [np.nan]*len(pfits_CS[ii])
        else:
            print('Function Min in simulation run was unsuccessful.')

        func_result = minimize(func_tomin, p0, args=[y_it, xdata, sigma], bounds=bounds2)
        if func_result['success']:   
            pfits_func[ii] = func_result['x']
            fbounds_met = np.array([bounds2[ii][0]<=func_result['x'][ii]<=bounds2[ii][1] for ii in np.arange(len(func_result['x']))]).all()
            if ~fbounds_met:   # check if fit parameters are not within bounds
                print('Some or all function fit parameters returned were not within the prescribed bounds. \n Changing these to NaNs. \n')
                pfits_func[ii] = [np.nan]*len(pfits_func[ii])
        else:
            print('Function Min in simulation run was unsuccessful.')

    if show_yplots:
        for yy, yit in enumerate(y_its.T):   # check simulated ydata is a normal dist'n
            plt.figure()
            plt.hist(yit, bins=20, label='Simulated Data')
            plt.axvline(ydata[yy], color='k', linestyle='dashed', label='Measured Value')
            plt.legend()
            plt.title(bolos[yy])
            plt.annotate(r'N$_{iterations}$ = %d'%num_its, (min(yit), 10))
            if save_figs: plt.savefig(plot_dir + bolos[yy] + '_simydata.png', dpi=300) 
        
    # sort results
    LS_params = np.mean(pfits_LS, axis=0); CS_params = np.mean(pfits_CS, axis=0); func_params = np.mean(pfits_func, axis=0)
    LS_std = np.std(pfits_LS, axis=0); CS_std = np.std(pfits_CS, axis=0); func_std = np.std(pfits_func, axis=0)

    # print results
    U_LSMC, W_LSMC, I_LSMC, aU_LSMC, aW_LSMC, aI_LSMC = LS_params   # parameter fits from Monte Carlo Least Squares Minimization
    Uerr_LSMC, Werr_LSMC, Ierr_LSMC, aUerr_LSMC, aWerr_LSMC, aIerr_LSMC = LS_std   # parameter errors from Monte Carlo Least Squares Minimization
    U_CSMC, W_CSMC, I_CSMC, aU_CSMC, aW_CSMC, aI_CSMC = CS_params   # parameter fits from Monte Carlo Chi-Squared Minimization
    Uerr_CSMC, Werr_CSMC, Ierr_CSMC, aUerr_CSMC, aWerr_CSMC, aIerr_CSMC = CS_std   # parameter errors from Monte Carlo Chi-Squared Minimization
    U_funcMC, W_funcMC, I_funcMC, aU_funcMC, aW_funcMC, aI_funcMC = func_params   # parameter fits from Monte Carlo Function Minimization
    Uerr_funcMC, Werr_funcMC, Ierr_funcMC, aUerr_funcMC, aWerr_funcMC, aIerr_funcMC = func_std   # parameter errors from Monte Carlo Function Minimization

    print('')   # least-squared minimization results
    print('Results from LSM')
    print('G_SiN(300 nm) = ', round(U_LS, 2), ' +/- ', round(Uerr_LS, 2))
    print('G_W(400 nm) = ', round(W_LS, 2), ' +/- ', round(Werr_LS, 2))
    print('G_I(400 nm) = ', round(I_LS, 2), ' +/- ', round(Ierr_LS, 2))
    print('alpha_U = ', round(aU_LS, 2), ' +/- ', round(aUerr_LS, 2))
    print('alpha_W = ', round(aW_LS, 2), ' +/- ', round(aWerr_LS, 2))
    print('alpha_I = ', round(aI_LS, 2), ' +/- ', round(aIerr_LS, 2))
    print('')
    print('Results from Monte Carlo sim - LSM')
    print('G_SiN(300 nm) = ', round(U_LSMC, 2), ' +/- ', round(Uerr_LSMC, 2))
    print('G_W(400 nm) = ', round(W_LSMC, 2), ' +/- ', round(Werr_LSMC, 2))
    print('G_I(400 nm) = ', round(I_LSMC, 2), ' +/- ', round(Ierr_LSMC, 2))
    print('alpha_U = ', round(aU_LSMC, 2), ' +/- ', round(aUerr_LSMC, 2))
    print('alpha_W = ', round(aW_LSMC, 2), ' +/- ', round(aWerr_LSMC, 2))
    print('alpha_I = ', round(aI_LSMC, 2), ' +/- ', round(aIerr_LSMC, 2))
    print('')

    if chi_result['success']:   # chi-squared results
        print('')   
        print('Results from Chi-Squared Min')
        print('G_SiN(300 nm) = ', round(U_CS, 2))
        print('G_W(400 nm) = ', round(W_CS, 2))
        print('G_I(400 nm) = ', round(I_CS, 2))
        print('alpha_U = ', round(aU_CS, 2))
        print('alpha_W = ', round(aW_CS, 2))
        print('alpha_I = ', round(aI_CS, 2))
    else:
        print('Chi-Squared Min (Single Run) was unsuccessful.')
    print('')
    print('Results from Monte Carlo sim - CSM')
    print('G_SiN(300 nm) = ', round(U_CSMC, 2), ' +/- ', round(Uerr_CSMC, 2))
    print('G_W(400 nm) = ', round(W_CSMC, 2), ' +/- ', round(Werr_CSMC, 2))
    print('G_I(400 nm) = ', round(I_CSMC, 2), ' +/- ', round(Ierr_CSMC, 2))
    print('alpha_U = ', round(aU_CSMC, 2), ' +/- ', round(aUerr_CSMC, 2))
    print('alpha_W = ', round(aW_CSMC, 2), ' +/- ', round(aWerr_CSMC, 2))
    print('alpha_I = ', round(aI_CSMC, 2), ' +/- ', round(aIerr_CSMC, 2))
    print('')

    if func_result['success']:       
        print('')   # function results (also chi-squared?)
        print('Results from Hand-Written Function Min')
        print('G_SiN(300 nm) = ', round(U_func, 2))
        print('G_W(400 nm) = ', round(W_func, 2))
        print('G_I(400 nm) = ', round(I_func, 2))
        print('alpha_U = ', round(aU_func, 2))
        print('alpha_W = ', round(aW_func, 2))
        print('alpha_I = ', round(aI_func, 2))
    else:
        print('Function Min (Single Run) was unsuccessful.')
    print('')
    print('Results from Monte Carlo sim - Func Min')
    print('G_SiN(300 nm) = ', round(U_funcMC, 2), ' +/- ', round(Uerr_funcMC, 2))
    print('G_W(400 nm) = ', round(W_funcMC, 2), ' +/- ', round(Werr_funcMC, 2))
    print('G_I(400 nm) = ', round(I_funcMC, 2), ' +/- ', round(Ierr_funcMC, 2))
    print('alpha_U = ', round(aU_funcMC, 2), ' +/- ', round(aUerr_funcMC, 2))
    print('alpha_W = ', round(aW_funcMC, 2), ' +/- ', round(aWerr_funcMC, 2))
    print('alpha_I = ', round(aI_funcMC, 2), ' +/- ', round(aIerr_funcMC, 2))
    print('')

    return LS_params, LS_std, CS_params, CS_std, func_params, func_std

def run_sim_justfunc(num_its, init_guess, ydata, xdata, ysigma, bounds1, bounds2, show_yplots=False, save_figs=False):   ### MC sim of hand-written function minimization

    print('Running minimization simulation over hand-written function.')
    func_result = minimize(func_tomin, init_guess, args=[ydata, xdata, ysigma], bounds=bounds2) 
    if func_result['success']:       
        U_func, W_func, I_func, aU_func, aW_func, aI_func = func_result['x']
    else:
        print('Function Min was unsuccessful.'); return 

    pfits_func = np.empty((num_its, len(init_guess)))
    y_its = np.empty((num_its, len(ydata)))
    for ii in np.arange(num_its):   # run simulation
        y_its[ii] = np.random.normal(ydata, ysigma)   # pull G's from normal distribution characterized by fit error
        func_result = minimize(func_tomin, init_guess, args=[y_its[ii], xdata, ysigma], bounds=bounds2)
        pfits_func[ii] = func_result['x']

    if show_yplots:
        for yy, yit in enumerate(y_its.T):   # check simulated ydata is a normal dist'n
            plt.figure()
            plt.hist(yit, bins=20, label='Simulated Data')
            plt.axvline(ydata[yy], color='k', linestyle='dashed', label='Measured Value')
            plt.legend()
            plt.title(bolos[yy])
            plt.annotate(r'N$_{iterations}$ = %d'%num_its, (min(yit), 10))
            if save_figs: plt.savefig(plot_dir + bolos[yy] + '_simydata.png', dpi=300) 
        
    # sory & print results    
    func_params = np.mean(pfits_func, axis=0); func_std = np.std(pfits_func, axis=0)
    U_funcMC, W_funcMC, I_funcMC, aU_funcMC, aW_funcMC, aI_funcMC = func_params   # parameter fits from Monte Carlo Function Minimization
    Uerr_funcMC, Werr_funcMC, Ierr_funcMC, aUerr_funcMC, aWerr_funcMC, aIerr_funcMC = func_std   # parameter errors from Monte Carlo Function Minimization

    print('')   # function results 
    print('Results from Hand-Written Chi-Squared Min')
    print('G_SiN(300 nm) = ', round(U_func, 2))
    print('G_W(400 nm) = ', round(W_func, 2))
    print('G_I(400 nm) = ', round(I_func, 2))
    print('alpha_U = ', round(aU_func, 2))
    print('alpha_W = ', round(aW_func, 2))
    print('alpha_I = ', round(aI_func, 2))
    print('')
    print('Results from Monte Carlo sim - H-W func Min')
    print('G_SiN(300 nm) = ', round(U_funcMC, 2), ' +/- ', round(Uerr_funcMC, 2))
    print('G_W(400 nm) = ', round(W_funcMC, 2), ' +/- ', round(Werr_funcMC, 2))
    print('G_I(400 nm) = ', round(I_funcMC, 2), ' +/- ', round(Ierr_funcMC, 2))
    print('alpha_U = ', round(aU_funcMC, 2), ' +/- ', round(aUerr_funcMC, 2))
    print('alpha_W = ', round(aW_funcMC, 2), ' +/- ', round(aWerr_funcMC, 2))
    print('alpha_I = ', round(aI_funcMC, 2), ' +/- ', round(aIerr_funcMC, 2))
    print('')
    return func_params, func_std

def redfunc_tomin(params, p0, data, param):   # reduced function to minimize
    
    GU0, GW0, GI0, aU0, aW0, aI0 = p0

    if param == 'U': params_topass = [params[0], GW0, GI0, params[1], aW0, aI0]
    elif param == 'W': params_topass = [GU0, params[0], GI0, aU0, params[1], aI0]
    elif param == 'I': params_topass = [GU0, GW0, params[0], aU0, aW0, params[1]]

    return func_tomin(params_topass, data)

# def redfunc_G(params, p0, data, param):   # reduced function to minimize
    
#     GU0, GW0, GI0, aU0, aW0, aI0 = p0

#     if param == 'UW': params_topass = [params[0], params[1], GI0, aU0, aW0, aI0]
#     elif param == 'UI': params_topass = [params[0], GW0, params[1], aU0, aW0, aI0]
#     elif param == 'WI': params_topass = [GU0, params[0], params[1], aU0, aW0, aI0]

#     return func_tomin(params_topass, data)

def runsim_red(num_its, p0, ydata, xdata, ysigma, bounds, param, show_yplots=False, save_figs=False):   ### MC sim of hand-written function minimization

    print('\n'); print('Running Minimization Simulation'); print('\n')

    if param == 'U': p0red = p0[[0,3]]
    elif param == 'W': p0red = p0[[1,4]]
    elif param == 'I': p0red = p0[[2,5]]

    func_result = minimize(redfunc_tomin, p0red, args=(p0, [ydata, xdata, sigma], param), bounds=bounds)
    if func_result['success']:  
        if bounds[0][0] <= func_result['x'][0] <=  bounds[0][1] and bounds[1][0] <= func_result['x'][1] <=  bounds[1][1]:   # check fit result is within bounds
            G_func, a_func = func_result['x']
        else:
            print('Single inimization returned result outside of bounds.')
            G_func, a_func = [np.nan, np.nan]
    else:
        print('Function Min was unsuccessful.'); return 

    pfits_func = np.empty((num_its, 2))
    y_its = np.empty((num_its, len(ydata)))
    for ii in np.arange(num_its):   # run simulation
        y_its[ii] = np.random.normal(ydata, ysigma)   # pull G's from normal distribution characterized by fit error
        # func_result = minimize(redfunc_tomin, p0red, args=(p0, [ydata, xdata, sigma], param), bounds=bounds)
        func_result = minimize(redfunc_tomin, p0red, args=(p0, [y_its[ii], xdata, sigma], param), bounds=bounds)
        if bounds[0][0] <= func_result['x'][0] <=  bounds[0][1] and bounds[1][0] <= func_result['x'][1] <=  bounds[1][1]:   # check fit result is within bounds
            pfits_func[ii] = func_result['x']
        else:
            print('Minimization in simulation returned result outside of bounds.')
            pfits_func[ii] = [np.nan, np.nan]
        
    if show_yplots:
        for yy, yit in enumerate(y_its.T):   # check simulated ydata is a normal dist'n
            plt.figure()
            plt.hist(yit, bins=20, label='Simulated Data')
            plt.axvline(ydata[yy], color='k', linestyle='dashed', label='Measured Value')
            plt.legend()
            plt.title(bolos[yy])
            plt.annotate(r'N$_{iterations}$ = %d'%num_its, (min(yit), 10))
            if save_figs: plt.savefig(plot_dir + bolos[yy] + '_simydata.png', dpi=300) 
        
    # sory & print results    
    func_params = np.mean(pfits_func, axis=0); func_std = np.std(pfits_func, axis=0)
    G_MC, a_MC = func_params; Gerr_MC, aerr_MC = func_std    # parameter fits / errors from Monte Carlo Function Minimization

    print('')   # function results 
    print('Results from Hand-Written Chi-Squared Min')
    print('G', param, ' = ', round(G_func, 2))
    print('alpha', param, ' = ', round(a_func, 2))
    print('')
    print('Results from Monte Carlo sim - H-W func Min')
    print('G', param, ' = ', round(G_MC, 2), ' +/- ', round(Gerr_MC, 2))
    print('alpha_', param, ' = ', round(a_MC, 2), ' +/- ', round(aerr_MC, 2))
    print('')
    return func_params, func_std

def Gfunc(params, p0, data, param):   # reduce params to 1 G value
        
    ydata, xdata, sigma = data
    GU0, GW0, GI0, aU0, aW0, aI0 = p0
    data_topass = [ydata, xdata, sigma]

    if param == 'U': params_topass = [params, GW0, GI0, aU0, aW0, aI0]
    elif param == 'W': params_topass = [GU0, params, GI0, aU0, aW0, aI0]
    elif param == 'I': params_topass = [GU0, GW0, params, aU0, aW0, aI0]

    return func_tomin(params_topass, data_topass)

def afunc(params, p0, data, param):   # reduce params to 1 alpha value
    
    ydata, xdata, sigma = data
    GU0, GW0, GI0, aU0, aW0, aI0 = p0
    data_topass = [ydata, xdata, sigma]

    if param == 'U': params_topass = [GU0, GW0, GI0, params, aW0, aI0]
    elif param == 'W': params_topass = [GU0, GW0, GI0, aU0, params, aI0]
    elif param == 'I': params_topass = [GU0, GW0, GI0, aU0, aW0, params]

    return func_tomin(params_topass, data_topass)
   

# def runsimred_G(num_its, p0, ydata, xdata, ysigma, bounds, param, show_yplots=False, save_figs=False):   ### MC sim of hand-written function minimization

#     print('\n'); print('Running Minimization Simulation'); print('\n')

#     if param == 'UW': p0red = p0[[0,1]]
#     elif param == 'UI': p0red = p0[[0,2]]
#     elif param == 'WI': p0red = p0[[1,2]]

#     func_result = minimize(redfunc_G, p0red, args=(p0, [ydata, xdata, sigma], param), bounds=bounds)
#     if func_result['success']:       
#         G1_func, G2_func = func_result['x']
#     else:
#         print('Function Min was unsuccessful.'); return

#     pfits_func = np.empty((num_its, 2))
#     y_its = np.empty((num_its, len(ydata)))
#     for ii in np.arange(num_its):   # run simulation
#         y_its[ii] = np.random.normal(ydata, ysigma)   # pull G's from normal distribution characterized by fit error
#         # func_result = minimize(redfunc_G, p0red, args=(p0, [ydata, xdata, sigma], param), bounds=bounds)
#         func_result = minimize(redfunc_G, p0red, args=(p0, [y_its[ii], xdata, sigma], param), bounds=bounds)
#         pfits_func[ii] = func_result['x']

#     if show_yplots:
#         for yy, yit in enumerate(y_its.T):   # check simulated ydata is a normal dist'n
#             plt.figure()
#             plt.hist(yit, bins=20, label='Simulated Data')
#             plt.axvline(ydata[yy], color='k', linestyle='dashed', label='Measured Value')
#             plt.legend()
#             plt.title(bolos[yy])
#             plt.annotate(r'N$_{iterations}$ = %d'%num_its, (min(yit), 10))
#             if save_figs: plt.savefig(plot_dir + bolos[yy] + '_simydata.png', dpi=300) 
        
#     # sory & print results    
#     func_params = np.mean(pfits_func, axis=0); func_std = np.std(pfits_func, axis=0)
#     G1_MC, G2_MC = func_params; G1err_MC, G2err_MC = func_std    # parameter fits / errors from Monte Carlo Function Minimization

#     print('')   # function results 
#     print('Results from Single Min')
#     print('G', param[0], ' = ', round(G1_func, 2))
#     print('G', param[1], ' = ', round(G2_func, 2))
#     print('')
#     print('Results from Monte Carlo sim')
#     print('G', param[0], ' = ', round(G1_MC, 2), ' +/- ', round(G1err_MC, 2))
#     print('G', param[1], ' = ', round(G2_MC, 2), ' +/- ', round(G2err_MC, 2))
#     print('')
#     return func_params, func_std
    


### Execute Analysis
if run_sim_allthree:   # run simulation for all three minimization routines
    LS_params, LS_std, CS_params, CS_std, func_params, func_std = run_sim(n_its, p0, ydata, xdata, sigma, bounds1, bounds2, show_yplots=show_plots, save_figs=save_figs)

if run_testsim:   # run simulation with just hand-written function
    func_params_test, func_std_test = run_sim_justfunc(n_its, p0, ydata, xdata, sigma, bounds1, bounds2, show_yplots=show_plots, save_figs=save_figs)
    func_params_test_J, func_std_test_J = run_sim_justfunc(n_its, p0, ydata_J, xdata, sigma_J, bounds1, bounds2, show_yplots=show_plots, save_figs=save_figs)

if random_initguess:
    iguess_bounds = np.array([2, 2, 2, 2, 2, 2])   # upper limit on guess
    iguesses = np.empty((num_guesses, len(iguess_bounds)))
    ls_params = np.empty((num_guesses, len(iguess_bounds))); cs_params = np.empty((num_guesses, len(iguess_bounds)))
    ls_std = np.empty((num_guesses, len(iguess_bounds))); cs_std = np.empty((num_guesses, len(iguess_bounds)))
    for ii in np.arange(num_guesses):
        iguesses[ii] = np.random.uniform(size=6)*iguess_bounds   # pull an inital guess between 0 and upper bound from normal dist'n
        ls_params[ii], ls_std[ii], cs_params[ii], cs_std[ii] = run_sim(n_its, iguesses[ii], ydata, xdata, sigma, bounds1, bounds2)

if quality_plots:

    xgridlim=[0,3]; ygridlim=[0,2]   # alpha_layer vs G_layer 
    xgrid, ygrid = np.mgrid[xgridlim[0]:xgridlim[1]:100j, ygridlim[0]:ygridlim[1]:100j]
    gridparams_U = np.array([xgrid, p0[1]*np.ones_like(xgrid), p0[2]*np.ones_like(xgrid), ygrid, p0[4]*np.ones_like(ygrid), p0[5]*np.ones_like(ygrid)]).T
    gridparams_W = np.array([p0[0]*np.ones_like(xgrid), xgrid, p0[2]*np.ones_like(xgrid), p0[3]*np.ones_like(ygrid), ygrid, p0[5]*np.ones_like(ygrid)]).T
    gridparams_I = np.array([p0[0]*np.ones_like(xgrid), p0[1]*np.ones_like(xgrid), xgrid, p0[3]*np.ones_like(ygrid), p0[4]*np.ones_like(ygrid), ygrid]).T

    chigrid_U = calc_chisq_grid(gridparams_U, [ydata, xdata, sigma])
    chigrid_W = calc_chisq_grid(gridparams_W, [ydata, xdata, sigma])
    chigrid_I = calc_chisq_grid(gridparams_I, [ydata, xdata, sigma])

    plt.figure()   # U vs aU parameter space
    im = plt.imshow(chigrid_U, cmap=plt.cm.RdBu, vmin=0, vmax=1E3, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower') 
    plt.colorbar(im)
    plt.title('Chi^2')
    plt.xlabel('U')
    plt.ylabel('a$_U$')
    if save_figs: plt.savefig(plot_dir + 'qualityplotU.png', dpi=300) 

    plt.figure()   # W vs aW parameter space
    im = plt.imshow(chigrid_W, cmap=plt.cm.RdBu, vmin=0, vmax=1E3, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower') 
    plt.colorbar(im)
    plt.title('Chi^2')
    plt.xlabel('W')
    plt.ylabel('a$_W$')
    if save_figs: plt.savefig(plot_dir + 'qualityplotW.png', dpi=300) 

    plt.figure()   # I vs aI parameter space
    im = plt.imshow(chigrid_I, cmap=plt.cm.RdBu, vmin=0, vmax=1E3, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower') 
    plt.colorbar(im)
    plt.title('Chi^2')
    plt.xlabel('I')
    plt.ylabel('a$_I$')
    if save_figs: plt.savefig(plot_dir + 'qualityplotI.png', dpi=300) 

    funcgrid_U = calc_func_grid(gridparams_U, [ydata, xdata, sigma])
    funcgrid_W = calc_func_grid(gridparams_W, [ydata, xdata, sigma])
    funcgrid_I = calc_func_grid(gridparams_I, [ydata, xdata, sigma])

    funcresult_U = minimize(redfunc_tomin, p0[[0,3]], args=(p0, [ydata, xdata, sigma], 'U'), bounds=boundsred) 
    if funcresult_U['success']:       
        U_func, aU_func = funcresult_U['x']
    else:
        print('U Min was unsuccessful.')

    funcresult_W = minimize(redfunc_tomin, p0[[1,4]], args=(p0, [ydata, xdata, sigma], 'W'), bounds=boundsred)     
    if funcresult_W['success']:       
        W_func, aW_func = funcresult_W['x']
    else:
        print('W Min was unsuccessful.')

    funcresult_I = minimize(redfunc_tomin, p0[[2,5]], args=(p0, [ydata, xdata, sigma], 'I'), bounds=boundsred) 
    if funcresult_I['success']:       
        I_func, aI_func = funcresult_I['x']
    else:
        print('Func Min was unsuccessful.')

    plt.figure()   # U vs aU parameter space
    im = plt.imshow(funcgrid_U, cmap=plt.cm.RdBu, vmin=0, vmax=1E3, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower') 
    if funcresult_U['success']: plt.plot(U_func, aU_func, 'mx', label='Func Min - 1 Run')
    plt.colorbar(im)
    plt.title('Hand-Written Function')
    plt.xlabel('U')
    plt.ylabel('a$_U$')
    if save_figs: plt.savefig(plot_dir + 'qualityplotU_hwfunc.png', dpi=300) 

    plt.figure()   # W vs aW parameter space
    im = plt.imshow(funcgrid_W, cmap=plt.cm.RdBu, vmin=0, vmax=1E3, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower') 
    if funcresult_W['success']: plt.plot(W_func, aW_func, 'mx', label='Func Min - 1 Run')
    plt.colorbar(im)
    plt.title('Hand-Written Function')
    plt.xlabel('W')
    plt.ylabel('a$_W$')
    if save_figs: plt.savefig(plot_dir + 'qualityplotW_hwfunc.png', dpi=300) 

    plt.figure()   # I vs aI parameter space
    im = plt.imshow(funcgrid_I, cmap=plt.cm.RdBu, vmin=0, vmax=1E3, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower') 
    if funcresult_I['success']: plt.plot(I_func, aI_func, 'mx', label='Func Min - 1 Run')
    plt.colorbar(im)
    plt.title('Hand-Written Function')
    plt.xlabel('I')
    plt.ylabel('a$_I$')
    if save_figs: plt.savefig(plot_dir + 'qualityplotI_hwfunc.png', dpi=300) 

if reduce_min:   # troubleshoot minimization by reducing number of parameters

    def overplot_qp(p0, data, boundsred, n_its, fits_J, param, full_res=[], savefigs=False):

        if param == 'U' or param=='W' or param=='I':
            xgridlim=[0,3]; ygridlim=[0,2]   # alpha_layer vs G_layer 
            xgrid, ygrid = np.mgrid[xgridlim[0]:xgridlim[1]:100j, ygridlim[0]:ygridlim[1]:100j]
            xlab = 'G'+param  
            if param=='U': 
                pinds = [0,3]
                gridparams = np.array([xgrid, p0[1]*np.ones_like(xgrid), p0[2]*np.ones_like(xgrid), ygrid, p0[4]*np.ones_like(ygrid), p0[5]*np.ones_like(ygrid)]).T
                ylab = 'a$_U$'
            elif param=='W': 
                pinds = [1,4]
                gridparams = np.array([p0[0]*np.ones_like(xgrid), xgrid, p0[2]*np.ones_like(xgrid), p0[3]*np.ones_like(ygrid), ygrid, p0[5]*np.ones_like(ygrid)]).T
                ylab = 'a$_W$'
            elif param=='I': 
                pinds = [2,5]
                gridparams = np.array([p0[0]*np.ones_like(xgrid), p0[1]*np.ones_like(xgrid), xgrid, p0[3]*np.ones_like(ygrid), p0[4]*np.ones_like(ygrid), ygrid]).T
                ylab = 'a$_I$'
            
            p0red = p0[pinds]
            res_single= minimize(redfunc_tomin, p0red, args=(p0, data, param), bounds=boundsred)   # Single Min 
            if res_single['success']:       
                x1_single, x2_single = res_single['x']
            else:
                print(param + 'Min was unsuccessful.')

            MC_params, MC_std = runsim_red(n_its, p0, ydata, xdata, sigma, boundsred, param)   # MC Sim 
            x1_MC, x2_MC = MC_params; x1err_MC, x2err_MC =  MC_std    # parameter fits / errors from Monte Carlo Function Minimization

        # elif param == 'UW' or param=='UI' or param=='WI':
        #     xgridlim=[0,3]; ygridlim=[0,3]   # G_layer vs G_layer 
        #     xgrid, ygrid = np.mgrid[xgridlim[0]:xgridlim[1]:100j, ygridlim[0]:ygridlim[1]:100j]  
        #     xlab = 'G'+param[0]
        #     ylab = 'G'+param[1]
        #     if param=='UW': 
        #         pinds = [0,1]
        #         gridparams = np.array([xgrid, ygrid, p0[2]*np.ones_like(xgrid), p0[3]*np.ones_like(ygrid), p0[4]*np.ones_like(ygrid), p0[5]*np.ones_like(ygrid)]).T
        #     elif param=='UI': 
        #         pinds = [0,2]
        #         gridparams = np.array([xgrid, p0[1]*np.ones_like(xgrid), ygrid, p0[3]*np.ones_like(ygrid), p0[4]*np.ones_like(ygrid), p0[5]*np.ones_like(ygrid)]).T
        #     elif param=='WI': 
        #         pinds = [1,2]
        #         gridparams = np.array([p0[0]*np.ones_like(ygrid), xgrid, ygrid, p0[3]*np.ones_like(ygrid), p0[4]*np.ones_like(ygrid), p0[5]*np.ones_like(ygrid)]).T

        #     p0red = p0[pinds]
        #     res_single = minimize(redfunc_G, p0red, args=(p0, data, param), bounds=boundsred) 
        #     if res_single['success']:       
        #         x1_single, x2_single = res_single['x']
        #     else:
        #         print(param + 'Min was unsuccessful.')

        #     MC_params, MC_std = runsimred_G(n_its, p0, ydata, xdata, sigma, boundsred, param)   # MC Sim 
        #     x1_MC, x2_MC = MC_params; x1err_MC, x2err_MC =  MC_std   # parameter fits / errors from Monte Carlo Function Minimization
    
        else:
            # print('Invalid parameter choice. Available choices: U, W, I, UW, UI, WI')
            print('Invalid parameter choice. Available choices: U, W, I')

        funcgrid = calc_func_grid(gridparams, data)   # Grid for Quality Plots
        Jres = fits_J[pinds]   # Joe's results
            
        plt.figure()   # U vs aU parameter space
        im = plt.imshow(funcgrid, cmap=plt.cm.RdBu, vmin=0, vmax=1E3, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower', alpha=0.8) 
        plt.errorbar(x1_MC, x2_MC, xerr=x1err_MC, yerr=x2err_MC, color='forestgreen', label='MC Sim', capsize=1)   # matching data and model colors
        if list(full_res): plt.errorbar(full_res[0,pinds[0]], full_res[0,pinds[1]], xerr=full_res[1, pinds[0]], yerr=full_res[1, pinds[1]], color='aqua', label='Full MC Sim', capsize=1)   # matching data and model colors
        if res_single['success']: plt.plot(x1_single, x2_single, 'x', mew=1.3, color='k', label='Single Min')
        plt.plot(Jres[0], Jres[1], 'x', mew=1.3, color='darkviolet', label="Joel's Results")
        plt.colorbar(im)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()
        plt.xlim(xgridlim[0], xgridlim[1]); plt.ylim(ygridlim[0], ygridlim[1])
        if save_figs: plt.savefig(plot_dir + 'redqualityplot_' + param + '.png', dpi=300)

        return res_single['x'], MC_params, MC_std

    LS_params, LS_std, CS_params, CS_std, func_params, func_std = run_sim(n_its, p0, ydata, xdata, sigma, bounds1, bounds2, show_yplots=show_plots, save_figs=save_figs)
    full_res = np.array([func_params, func_std])   # include full MC simulation results on QPs
    U_single, U_MC, Uerr_MC = overplot_qp(p0, data, boundsred, n_its, fits_J, 'U', full_res=full_res, savefigs=save_figs)
    W_single, W_MC, Werr_MC = overplot_qp(p0, data, boundsred, n_its, fits_J, 'W', full_res=full_res, savefigs=save_figs)
    I_single, I_MC, Ierr_MC = overplot_qp(p0, data, boundsred, n_its, fits_J, 'I', full_res=full_res, savefigs=save_figs)
    # UW_single, UW_MC, UWerr_MC = overplot_qp(p0, data, boundsred, n_its, fits_J, 'UW', savefigs=save_figs)
    # UI_single, UI_MC, UIerr_MC = overplot_qp(p0, data, boundsred, n_its, fits_J, 'UI', savefigs=save_figs)
    # WI_single, WI_MC, WIerr_MC = overplot_qp(p0, data, boundsred, n_its, fits_J, 'WI', savefigs=save_figs)


if single_param:   # troubleshoot minimization by reducing number of parameters

    # Gbounds = (0, 3); abounds = (0, 2)
    Glim = (0, 3); alim = (0, 1)
    Grange = np.linspace(Glim[0], Glim[1]); arange = np.linspace(alim[0], alim[1])
    data = [ydata, xdata, sigma]
   
    GUfunc = np.array([Gfunc(GU, p0, data, 'U') for GU in Grange])
    funcresult_U = minimize_scalar(Gfunc, args=(p0, data, 'U'), bounds=boundsred[0]) 
    U_func = funcresult_U['x'] if funcresult_U['success'] else print('U Min was unsuccessful.')

    plt.figure()
    plt.plot(Grange, GUfunc, label='Function')
    plt.axvline(U_func, label='Min GU Value', color='C1')
    plt.legend()
    plt.xlabel('G_U [pW/K]')
    plt.ylabel('Function to Minimize')
        
    GWfunc = np.array([Gfunc(GW, p0, data, 'W') for GW in Grange])
    funcresult_W = minimize_scalar(Gfunc, args=(p0, data, 'W'), bounds=boundsred[0]) 
    W_func = funcresult_W['x'] if funcresult_W['success'] else print('W Min was unsuccessful.')

    plt.figure()
    plt.plot(Grange, GWfunc, label='Function')
    plt.axvline(W_func, label='Min GW Value', color='C1')
    plt.legend()
    plt.xlabel('G_W [pW/K]')
    plt.ylabel('Function to Minimize')

    GIfunc = np.array([Gfunc(GI, p0, data, 'I') for GI in Grange])
    funcresult_I = minimize_scalar(Gfunc, args=(p0, data, 'I'), bounds=boundsred[0]) 
    I_func = funcresult_I['x'] if funcresult_I['success'] else print('I Min Ias unsuccessful.')

    plt.figure()
    plt.plot(Grange, GIfunc, label='Function')
    plt.axvline(I_func, label='Min GI Value', color='C1')
    plt.legend()
    plt.xlabel('G_I [pW/K]')
    plt.ylabel('Function to Minimize')

    aUfunc = np.array([afunc(aU, p0, data, 'U') for aU in arange])
    funcresult_aU = minimize_scalar(afunc, args=(p0, data, 'U'), bounds=boundsred[1]) 
    aU_func = funcresult_aU['x'] if funcresult_aU['success'] else print('aU Min was unsuccessful.')

    plt.figure()
    plt.plot(arange, aUfunc, label='Function')
    plt.axvline(aU_func, label='Min aU Value', color='C1')
    plt.legend()
    plt.xlabel('a_U')
    plt.ylabel('Function to Minimize')
        
    aWfunc = np.array([afunc(aW, p0, data, 'W') for aW in arange])
    funcresult_aW = minimize_scalar(afunc, args=(p0, data, 'W'), bracket=boundsred[1]) 
    aW_func = funcresult_aW['x'] if funcresult_aW['success'] else print('aW Min was unsuccessful.')

    plt.figure()
    plt.plot(arange, aWfunc, label='Function')
    plt.axvline(aW_func, label='Min aW Value', color='C1')
    plt.legend()
    plt.xlabel('a_W')
    plt.ylabel('Function to Minimize')

    aIfunc = np.array([afunc(aI, p0, data, 'I') for aI in arange])
    funcresult_aI = minimize_scalar(afunc, args=(p0, data, 'I'), bounds=boundsred[1]) 
    aI_func = funcresult_aI['x'] if funcresult_aI['success'] else print('aI Min was unsuccessful.')

    plt.figure()
    plt.plot(arange, aIfunc, label='Function')
    plt.axvline(aI_func, label='Min aI Value', color='C1')
    plt.legend()
    plt.xlabel('a_I')
    plt.ylabel('Function to Minimize')


plt.show()