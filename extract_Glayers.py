import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit, minimize, minimize_scalar
from bolotest_routines import *
# from scipy.stats import chi2

"""
Script for measuring individual layer contributions to the total thermal conductivity of a TES. 
Data from 'bolotest' measurement run in 2018. 
G_total and error is from TES power law fit at Tc, assuming dP/dT = G = n*k*Tc^(n-1)
Layers: U = SiN substrate, W = Nb wiring layers (<=2 layers per leg), I = SiN insulating layers (<=2 layers per leg)

aharkehosemann@gmail.com
2022/01

UPDATE 2022.06.15 : We recently discovered the wiring layers have a shorter width (W1=5um, W2=3um) than the SiN layers, and that there's a 120nm SiOx layer on the membrane under the 300nm SiNx layer. 
I now scale W2 terms by 3/5 and W stack by thickness-weighted average width (assumes G scales linearly with layer width). 
Shannon estimates 11-14 nm etching on the SiOx, so true thickness is ~109 nm. 

UPDATE 2022.06.28 : Should we model the data as five layers (treat substrate as 420 nm thick layer of one material) or six layers (one 120 nm SiOx and one 300nm layer SiNx)?
DOF of six-layer fit is 0...
Raw chi-squared value of five-layer is 32.0, six-layer is 34.6, so five-layer is technically better...
Not sure if looking at significance is useful because we aren't hypothesis testing and have very small DOF. 

TODO: look at bolo1b Gtotal values 
"""

### User Switches
# which analysis?
runsim_allthree = False   # do minimizaion routines n_its times
runsim_justfunc = False   # for debugging simulation
random_initguess = False   # try simulation with randomized initial guesses
quality_plots = True   # map G_x vs alpha_x space, plot full sim results + Joel's results + two-param results + single value results
average_qp = False   # show average value for 2D quality plot over MC sim to check for scattering
lit_compare = False
six_layers = True

# options
save_figs = True
show_plots = False   # show simulated y-data plots during MC simulation

n_its = 1000   # number of iterations for MC simulation
num_guesses = 100   # number of randomized initial guesses to try
plot_dir = '/Users/angi/NIS/Analysis/bolotest/plots/layer_extraction_analysis/'
fn_comments = '_170mK'   # filename comments
alim = [0, 2]   # limits for fitting alpha
# p0 = np.array([0.89, 0.3, 1.3, .58, 1, 1])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]; constrained alpha range
p0 = np.array([0.71, 0.69, 1.32, .15, 1.69, 1.25])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]; fit results with new w width and U thickness

# choose data set
# ydata = np.array([11.7145073, 4.921841228, 8.077815536, 10.03001622, 16.63099617, 5.386790491, 15.2863792, 3.585251305])   # pW/K at Tc, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13*; bolo1b is weighted average (not sure we trust row 10)
# sigma = np.array([0.100947739, 0.063601732, 0.078665632, 0.130040288, 0.142600818, 0.059261252, 0.123206779, 0.052084114])   # pW/K at Tc; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13*
# ydata_J = np.array([13.74, 4.921841228, 8.077815536, 10.03001622, 16.63099617, 5.386790491, 15.2863792, 3.585251305])   # pW/K at Tc, most were weighted averages*; bolo 1b, 24*, 23*, 22, 21*, 20*, 7*, 13*; only row 5 for bolo1b
# sigma_J = np.array([0.17, 0.063601732, 0.078665632, 0.130040288, 0.142600818, 0.059261252, 0.123206779, 0.052084114])   # pW/K at Tc; bolo 1b, 24*, 23*, 22, 21*, 20*, 7*, 13*
ydata_170mK = [13.74098652, 4.91314256, 8.006715762, 10.03001622, 16.49000297, 5.362145301, 15.14067701, 3.5772749]   # pW/K at 170 mK, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13* (only using row 5 value of bolo1b)
sigma_170mK = [0.171320173, 0.087731049, 0.108034223, 0.130040288, 0.19732244, 0.078047237, 0.166570199, 0.072651667]
ydata = ydata_170mK; sigma = sigma_170mK   

bolos = np.array(['bolo 1b', 'bolo 24', 'bolo 23', 'bolo 22', 'bolo 21', 'bolo 20', 'bolo 7', 'bolo 13'])
bounds1 = [(0, 0, 0, alim[0], alim[0], alim[0]), (np.inf, np.inf, np.inf, alim[1], alim[1], alim[1])]; bounds2 = [(0, np.inf), (0, np.inf), (0, np.inf), (alim[0], alim[1]), (alim[0], alim[1]), (alim[0], alim[1])]   # different fitting routines like different formats
fits_J = np.array([.89, 0.30, 1.3277, 0.58, 1.42, 1.200]); error_J = np.array([0.14, 0.18, 0.02, 0.57, 0.53, 0.06])   # compare with Joel's results
results_J = [fits_J, error_J]   # Joel's fit parameters and error
boundsred = [(0, np.inf), (alim[0], alim[1])]   # G, alpha

### Geometries
coeffs = np.array([[[4, 0, 0], [4, 4, 0], [4, 4, 0]],   # bolo 1b, consistent with Joel's numbers
        [[1, 3, 0], [1, 1, 0], [1, 1, 0]],   # bolo 24, consistent with Joel's numbers
        [[2, 2, 0], [2, 2, 0], [2, 2, 0]],   # bolo 23, consistent with Joel's numbers
        [[3, 1, 0], [3, 3, 0], [3, 3, 0]],   # bolo 22, consistent with Joel's numbers
        [[3, 1, 0], [1, 1, 0], [1, 3, 0]],   # bolo 21, consistent with Joel's numbers
        [[4, 0, 0], [3, 1, 1], [1, 0, 0]],   # bolo 20, consistent with Joel's numbers
        [[3, 1, 0], [3, 3, 0], [3, 1, 3]],   # bolo 7, consistent with Joel's numbers
        [[3, 1, 0], [1, 1, 0], [1, 0, 0]]])   # bolo 13, consistent with Joel's numbers

ratios = np.array([[[1, 0, 0], [160/400, 340/400, 0], [350/400, 1, 0]],   # bolo 1b, consistent with Joel's numbers
        [[1, 220/300, 0], [160/400, 340/400, 0], [350/400, 1, 0]],   # bolo 24, consistent with Joel's numbers
        [[1, 220/300, 0], [160/400, 340/400, 0], [350/400, 1, 0]],   # bolo 23, consistent with Joel's numbers
        [[1, 220/300, 0], [160/400, 340/400, 0], [350/400, 1, 0]],   # bolo 22, consistent with Joel's numbers
        [[280/300, 1, 0], [160/400, 285/400, 0], [350/400, (270+400)/400, 0]],   # bolo 21, consistent with Joel's numbers
        [[1, 0, 0], [(100+285)/400, 160/400, 285/400], [350/400, 0, 0]],   # bolo 20, consistent with Joel's numbers
        [[1, 280/300, 0], [160/400, 340/400, 0], [350/400, (270+400)/400, 1]],   # bolo 7, consistent with Joel's numbers
        [[220/300, 1, 0], [160/400, 285/400, 0], [350/400, 0, 0]]])   # bolo 13, consistent with Joel's numbers

xdata = np.array([coeffs, ratios])
data = [ydata, xdata, sigma] 

### Execute Analysis
if runsim_allthree:   # run simulation for all three minimization routines
    LS_params, LS_std, CS_params, CS_std, func_params, func_std = run_sim(n_its, p0, data, bounds1, bounds2, plot_dir, show_yplots=show_plots, fn_comments=fn_comments, save_figs=save_figs)

if runsim_justfunc:   # run simulation with just hand-written function
    func_params_test, func_std_test = runsim_func(n_its, p0, ydata, xdata, sigma, bounds1, bounds2, show_yplots=show_plots, save_figs=save_figs)
    func_params_test_J, func_std_test_J = runsim_func(n_its, p0, ydata_J, xdata, sigma_J, bounds1, bounds2, show_yplots=show_plots, save_figs=save_figs)

if random_initguess:
    iguess_bounds = np.array([2, 2, 2, 2, 2, 2])   # upper limit on guess
    iguesses = np.empty((num_guesses, len(iguess_bounds)))
    ls_params = np.empty((num_guesses, len(iguess_bounds))); cs_params = np.empty((num_guesses, len(iguess_bounds)))
    ls_std = np.empty((num_guesses, len(iguess_bounds))); cs_std = np.empty((num_guesses, len(iguess_bounds)))
    for ii in np.arange(num_guesses):
        iguesses[ii] = np.random.uniform(size=6)*iguess_bounds   # pull an inital guess between 0 and upper bound from normal dist'n
        ls_params[ii], ls_std[ii], cs_params[ii], cs_std[ii] = run_sim(n_its, iguesses[ii], ydata, xdata, sigma, bounds1, bounds2, plot_dir, fn_comments=fn_comments)

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
    if save_figs: plt.savefig(plot_dir + 'qualityplotU' + fn_comments + '.png', dpi=300) 

    plt.figure()   # W vs aW parameter space
    im = plt.imshow(chigrid_W, cmap=plt.cm.RdBu, vmin=0, vmax=1E3, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower') 
    plt.colorbar(im)
    plt.title('Chi^2')
    plt.xlabel('W')
    plt.ylabel('a$_W$')
    if save_figs: plt.savefig(plot_dir + 'qualityplotW' + fn_comments + '.png', dpi=300) 

    plt.figure()   # I vs aI parameter space
    im = plt.imshow(chigrid_I, cmap=plt.cm.RdBu, vmin=0, vmax=1E3, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower') 
    plt.colorbar(im)
    plt.title('Chi^2')
    plt.xlabel('I')
    plt.ylabel('a$_I$')
    if save_figs: plt.savefig(plot_dir + 'qualityplotI' + fn_comments + '.png', dpi=300) 

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
    if save_figs: plt.savefig(plot_dir + 'qualityplotU_hwfuncp0' + fn_comments + '.png', dpi=300) 

    plt.figure()   # W vs aW parameter space
    im = plt.imshow(funcgrid_W, cmap=plt.cm.RdBu, vmin=0, vmax=1E3, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower') 
    if funcresult_W['success']: plt.plot(W_func, aW_func, 'mx', label='Func Min - 1 Run')
    plt.colorbar(im)
    plt.title('Hand-Written Function')
    plt.xlabel('W')
    plt.ylabel('a$_W$')
    if save_figs: plt.savefig(plot_dir + 'qualityplotW_hwfuncp0' + fn_comments + '.png', dpi=300) 

    plt.figure()   # I vs aI parameter space
    im = plt.imshow(funcgrid_I, cmap=plt.cm.RdBu, vmin=0, vmax=1E3, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower') 
    if funcresult_I['success']: plt.plot(I_func, aI_func, 'mx', label='Func Min - 1 Run')
    plt.colorbar(im)
    plt.title('Hand-Written Function')
    plt.xlabel('I')
    plt.ylabel('a$_I$')
    if save_figs: plt.savefig(plot_dir + 'qualityplotI_hwfuncp0' + fn_comments + '.png', dpi=300) 

if quality_plots:   # plot G_x vs alpha_x parameter space with various fit results

    full_res = runsim_func(n_its, p0, data, bounds1, bounds2, plot_dir, show_yplots=show_plots, save_figs=save_figs, fn_comments=fn_comments)  
    params_fit, sigmas_fit =  full_res
    dof_six = len(ydata) - len(p0) - 1   # degrees of freedom for six-layer fit; number of data points - number of parameters - 1
    U_MC, Uerr_MC = overplot_qp(p0, data, boundsred, n_its, results_J, 'U', plot_dir, full_res=full_res, savefigs=save_figs, fn_comments=fn_comments, vmax=500)
    W_MC, Werr_MC = overplot_qp(p0, data, boundsred, n_its, results_J, 'W', plot_dir, full_res=full_res, savefigs=save_figs, fn_comments=fn_comments, vmax=500)
    I_MC, Ierr_MC = overplot_qp(p0, data, boundsred, n_its, results_J, 'I', plot_dir, full_res=full_res, savefigs=save_figs, fn_comments=fn_comments, vmax=500)

    Gmeas_U, Gmeas_W, Gmeas_I, alpham_U, alpham_W, alpham_I = params_fit; sigGU, sigGW, sigGI, sigalpha, sigalphaW, sigalphaI = sigmas_fit
    G_wirestack = Gmeas_W*(2/4)**(alpham_W+1) + Gmeas_I*(350/400)**alpham_I + 3/5*Gmeas_W + Gmeas_I
    print('G of full wire stack: ', round(G_wirestack, 2), ' pW / K')   # W1 + W2 + I1 + I2 as deposted; pW / K
    
    kappam_U = GtoKappa(Gmeas_U, 7*300E-3, L); sigkappam_SiN = GtoKappa(sigGU, 7*300E-3, L)   # pW / K / um
    kappam_W = GtoKappa(Gmeas_W, 5*400E-3, L); sigkappam_W = GtoKappa(sigGW, 5*400E-3, L)   # pW / K / um
    kappam_I = GtoKappa(Gmeas_I, 7*400E-3, L); sigkappam_I = GtoKappa(sigGI, 7*400E-3, L)   # pW / K / um
    print('Kappa_U: ', round(kappam_U, 2), ' +/- ', round(sigkappam_U, 2), ' pW/K um')
    print('Kappa_W: ', round(kappam_W, 2), ' +/- ', round(sigkappam_W, 2), ' pW/K um')
    print('Kappa_I: ', round(kappam_I, 2), ' +/- ', round(sigkappam_I, 2), ' pW/K um')

    chisq_fit = func_tomin(params_fit, data)
    print('Chi-squared for the five-layer fit: ', round(chisq_fit, 3))


if average_qp:   # check for scattering in full MC simulation vs single run

    def plot_meanqp(p0, data, n_its, param, plot_dir, savefigs=False, fn_comments=''):

        if param == 'U' or param=='W' or param=='I':
            xgridlim=[0,3]; ygridlim=[0,2]   # alpha_layer vs G_layer 
            xgrid, ygrid = np.mgrid[xgridlim[0]:xgridlim[1]:100j, ygridlim[0]:ygridlim[1]:100j]
            xlab = 'G'+param  
            if param=='U': 
                gridparams = np.array([xgrid, p0[1]*np.ones_like(xgrid), p0[2]*np.ones_like(xgrid), ygrid, p0[4]*np.ones_like(ygrid), p0[5]*np.ones_like(ygrid)]).T
                ylab = 'a$_U$'
            elif param=='W': 
                gridparams = np.array([p0[0]*np.ones_like(xgrid), xgrid, p0[2]*np.ones_like(xgrid), p0[3]*np.ones_like(ygrid), ygrid, p0[5]*np.ones_like(ygrid)]).T
                ylab = 'a$_W$'
            elif param=='I': 
                gridparams = np.array([p0[0]*np.ones_like(xgrid), p0[1]*np.ones_like(xgrid), xgrid, p0[3]*np.ones_like(ygrid), p0[4]*np.ones_like(ygrid), ygrid]).T
                ylab = 'a$_I$'

        funcgrid = calc_func_grid(gridparams, data)

        plt.figure()   # G vs alpha parameter space, single run 
        im = plt.imshow(funcgrid, cmap=plt.cm.RdBu, vmin=0, vmax=1E3, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower') 
        plt.colorbar(im)
        plt.title('Function Values in 2D Param Space')
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        if savefigs: plt.savefig(plot_dir + 'funcqualityplot_' + param + fn_comments + '.png', dpi=300)

        funcgridU_sim = np.empty((n_its, len(funcgrid[0]), len(funcgrid[1])))
        y_its = np.empty((n_its, len(ydata)))
        for ii in np.arange(n_its):   # run simulation
            y_its[ii] = np.random.normal(ydata, sigma)   # pull G's from normal distribution characterized by fit error
            data_it = [y_its[ii], xdata, sigma]   
            funcgridU_sim[ii] = calc_func_grid(gridparams, data_it)   # calculate function value with simulated y-data

        funcgrid_mean = np.mean(funcgridU_sim, axis=0)

        plt.figure()   # G vs alpha parameter space
        im = plt.imshow(funcgrid_mean, cmap=plt.cm.RdBu, vmin=0, vmax=1E3, extent=[min(xgridlim), max(xgridlim), min(ygridlim), max(ygridlim)], origin='lower', alpha=0.8) 
        plt.colorbar(im)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title('Mean Func Vals, N$_{its}$ = %d'%n_its) 
        # plt.xlim(xgridlim[0], xgridlim[1]); plt.ylim(ygridlim[0], ygridlim[1])
        if savefigs: plt.savefig(plot_dir + 'meanfuncqualityplot_' + param + fn_comments + '.png', dpi=300)
    
        return funcgrid, funcgrid_mean

    funcgrid_U, funcgridmean_U = plot_meanqp(p0, data, n_its, 'U', plot_dir, savefigs=save_figs, fn_comments=fn_comments)
    funcgrid_W, funcgridmean_W = plot_meanqp(p0, data, n_its, 'W', plot_dir, savefigs=save_figs, fn_comments=fn_comments)
    funcgrid_I, funcgridmean_I = plot_meanqp(p0, data, n_its, 'I', plot_dir, savefigs=save_figs, fn_comments=fn_comments)


if lit_compare:

    ### Nb
    kB = 1.3806503E-23   # Boltzmann constant, [J/K]
    NA = 6.022E23   # Avogadro's number
    hbar = 1.055E-34   # J s
    # molvol = 1.083E-5   # m^3 per mol
    L = 220   # TES leg length, um

    
    def kappa_permfp(T, TD_Nb, Tc, v, carrier=''):   # Leopold and Boorse 1964, Nb
        # INPUT: bath temp, Debye temperature, critical temperature, carrier velocity, mean free path
        # calculates theoretical thermal conductivity via specific heat for bulk Nb
        molvol = 1/(1.08E-5) * 1E-18  # mol per um^3 
        a = 8.21; b=1.52
        gamma = 7.8E-3   # J / mole / K^2, LEUPOLD & BOORSE 1964
        if carrier=='electron': C = (gamma*Tc*a*np.exp(-b*Tc/T) )/molvol*1E-18  # electron specific heat, electron from LEUPOLD & BOORSE 1964, J/K/um^3
        elif carrier=='phonon': C = ((12*np.pi**4*NA*kB)/5 * (T/TD_Nb)**3) * molvol  # phonon specific heat, J/K/um^3
        else: print('Invalid carrier; choose phonon or electron')
        return 1/3*C*v   # thermal conductivity via phonemenological gas kinetic theory / lambda, W / K / um^2

    def GtoKappa(G, A, L):   # thermal conductance, area, length
        return G*L/A

    Gmeas_U = 0.725E-12; sigGU = 0.07E-12   # W / K, results from SiN + SiO combined substrate layer
    Gmeas_W = 0.69E-12; sigGW = 0.14E-12   # our measured thermal conductance of 400nm film of Nb, width = 5um, W / K
    Gmeas_I = 1.32E-12; sigGI = 0.03E-12   # W / K
    alpham_U = 0.17; sigGU = 0.49   # W / K
    alpham_W = 1.68; sigGW = 0.56   # our measured thermal conductance of 400nm film of Nb, width = 5um, W / K
    alpham_I = 1.24; sigGI = 0.05   # W / K
    G_wirestack = Gmeas_W*(2/4)**(alpham_W+1) + Gmeas_I*(350/400)**alpham_I + 3/5*Gmeas_W + Gmeas_I

    ### Nb
    TD_Nb = 275   # K, Nb
    T = 0.170   # measurement temperature (Tc of TESs), K
    Tc = 9.2   # Nb, K
    vF = 1.37E6*1E6   # Nb Fermi velocity (electron velocity), um/s
    v_Nb = 3480*1E6   # phonon velocity is the speed of sound in Nb, um/s

    kappapmfp_Nb = np.sum([kappa_permfp(T, TD_Nb, Tc, vF, carrier='electron'), kappa_permfp(T, TD_Nb, Tc, v_Nb, carrier='phonon')])   # electron thermal conductivity + phonon thermal conductivity, W/K/m

    kappam_W = GtoKappa(Gmeas_W, 5*400E-3, L); sigkappam_W = GtoKappa(sigGW, 5*400E-3, L)
    print('Measured W Kappa: ', kappam_W*1E12, ' +/- ', sigkappam_W*1E12, ' pW/K um')
    print('Predicted W Kappa/mfp: ',  kappapmfp_Nb, ' W/K um^2')
    print('Measured W mfp: ', kappam_W/kappapmfp_Nb*1E3, ' nm')

    ### SiN
    kappam_I = GtoKappa(Gmeas_I, 7*400E-3, L); sigkappam_I = GtoKappa(sigGI, 7*400E-3, L)   # W / K um, x1E4 to get to W / K cm
    kappam_U = GtoKappa(Gmeas_U, 7*420E-3, L); sigkappam_U = GtoKappa(sigGU, 7*420E-3, L)   # W / K um, x1E4 to get to W / K cm
    normkappa_I = kappam_I / (np.sqrt(2)*.4)   # W / K um^2
    normkappa_U = kappam_I / (np.sqrt(2)*.3)   # W / K um^2
    print('Measured I Kappa: ', kappam_I*1E12, ' +/-', sigkappam_I*1E12, ' pW/K um')
    print('Measured U Kappa: ', kappam_U*1E12, '+/-', sigkappam_U*1E12, ' pW/K um')

    kappapmfp_wang = 1/3*(0.083*T+0.509*T**3)*6986 * 1E-12  # kappa per mfp W / um^2 / K; Si3N4; volume heat capacity * average sound speed
    kappa_wang = kappapmfp_wang*6.58   # W / K / um for 10 um width SiN beam
    G_wang = kappa_wang*640/(1*10)   # W / K for 640 um x 10 um x 1um SiN beam
    lambda_I = kappam_I/kappapmfp_wang   # um
    lambda_U = kappam_U/kappapmfp_wang   # um

    ### 2D vs 3D phonon dimensionality
    vt_SiN = 6.28E3   # transverse sound speed for Si3N4 at 0K, m / s, Bruls 2001
    Tdcrit_SiN = hbar * vt_SiN / kB   # K m
    Tdcrit_Nb = hbar * v_Nb*1E-6 / kB   # K m
    Td_400 = 0.170 * 400E-9   # K m
    Td_300 = 0.170 * 300E-9   # K m

    kappa_LP = 1.58*T**1.54*1E-9   # W / K um^2, Geometry III for SiN film from Leivo & Pekola 1998 (they say mfp = d)
    kappamfp_LP = 1.58*T**1.54*1E-9/0.200   # W / K um^2, Geometry III from Leivo & Pekola 1998 (they say mfp = d)


if six_layers:   # run simulation for substrate where SiOx and SiNx layer are separate
    # this yields 7 free parameters because SiOx layer is always 120 nm (no sensitivity to alpha_SiOx)

    L = 220   # TES leg length, um
    params_five = np.append(np.nan, full_res[0]); sigmas_five = np.append(np.nan, full_res[1])
    results_five = np.array([params_five, sigmas_five])  # five-layer parameters and error
    p0_six = np.array([0.15, 0.32, 0.49, 1.36, .15, 1.45, 1.19])   # SiO, SiN, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]
    bounds1_six = [(0, 0, 0, 0, alim[0], alim[0], alim[0]), (np.inf, np.inf, np.inf, np.inf, alim[1], alim[1], alim[1])]; bounds2_six = [(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (alim[0], alim[1]), (alim[0], alim[1]), (alim[0], alim[1])]   # different fitting routines like different formats
    dof_six = len(ydata) - len(p0_six) - 1   # degrees of freedom for six-layer fit; number of data points - number of parameters - 1

    fullres_six = runsim_sixlayers(n_its, p0_six, data, bounds1_six, bounds2_six, plot_dir, show_yplots=show_plots, save_figs=save_figs, fn_comments=fn_comments) 
    SiN_MC, SiNerr_MC = overplotqp_sixlayers(p0_six, data, boundsred, n_its, results_five, 'SiN', plot_dir, full_res=fullres_six, savefigs=save_figs, fn_comments=fn_comments+'_sixlayers', vmax=500)
    W_MC, Werr_MC = overplotqp_sixlayers(p0_six, data, boundsred, n_its, results_five, 'W', plot_dir, full_res=fullres_six, savefigs=save_figs, fn_comments=fn_comments+'_sixlayers', vmax=500)
    I_MC, Ierr_MC = overplotqp_sixlayers(p0_six, data, boundsred, n_its, results_five, 'I', plot_dir, full_res=fullres_six, savefigs=save_figs, fn_comments=fn_comments+'_sixlayers', vmax=500)
    
    params_six, sigmas_six = fullres_six
    Gmeas_SiO6, Gmeas_SiN6, Gmeas_W6, Gmeas_I6, alpham_U6, alpham_W6, alpham_I6 = params_six; sigGSiO6, sigGSiN6, sigGW6, sigGI6, sigalphaG6, sigalphaW6, sigalphaI6 = sigmas_six
    G_wirestack = Gmeas_W6*(2/4)**(alpham_W6+1) + Gmeas_I6*(350/400)**alpham_I6 + 3/5*Gmeas_W6 + Gmeas_I6
    print('G of full wire stack: ', round(G_wirestack, 2), ' pW / K')   # W1 + W2 + I1 + I2 as deposted; pW / K

    kappam_SiO6 = GtoKappa(Gmeas_SiO6, 7*109E-3, L); sigkappam_SiO6 = GtoKappa(sigGSiO6, 7*109E-3, L)   # pW / K / um
    kappam_SiN6 = GtoKappa(Gmeas_SiN6, 7*300E-3, L); sigkappam_SiN6 = GtoKappa(sigGSiN6, 7*300E-3, L)   # pW / K / um
    kappam_W6 = GtoKappa(Gmeas_W6, 5*400E-3, L); sigkappam_W6 = GtoKappa(sigGW6, 5*400E-3, L)   # pW / K / um
    kappam_I6 = GtoKappa(Gmeas_I6, 7*400E-3, L); sigkappam_I6 = GtoKappa(sigGI6, 7*400E-3, L)   # pW / K / um
    print('Kappa_SiO: ', round(kappam_SiO6, 2), ' +/- ', round(sigkappam_SiO6, 2), ' pW/K um')
    print('Kappa_SiN: ', round(kappam_SiN6, 2), ' +/- ', round(sigkappam_SiN6, 2), ' pW/K um')
    print('Kappa_W: ', round(kappam_W6, 2), ' +/- ', round(sigkappam_W6, 2), ' pW/K um')
    print('Kappa_I: ', round(kappam_I6, 2), ' +/- ', round(sigkappam_I6, 2), ' pW/K um')

    chisq_six = functomin_sixlayers(params_six, data)
    # print('Reduced chi-squared for six-layer fit: ', round(chisq_six/dof_six, 3))   # this is infinity because this model has 0 degrees of freedom
    print('Chi-squared for the six-layer fit: ', round(chisq_six, 3))



plt.show()