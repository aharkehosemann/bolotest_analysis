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

TODO: look at bolo1b Gtotal values, full W "MC Sim" error bars suspicious, error bars on Joel's numbers, do constrained alpha with Joel's numbers
"""

### User Switches
run_sim_allthree = False   # do minimizaion routines n_its times
run_testsim = False   # for debugging simulation
random_initguess = False   # try simulation with randomized initial guesses
reduce_min = True   # reduce minimization to reconcile quality plots with chi_sq min
average_qp = False   # show average value for 2D quality plot over MC sim to check for scattering
save_figs = False
show_plots = False   # show simulated y-data plots during MC simulation

n_its = 1000   # number of iterations for MC simulation
num_guesses = 100   # number of randomized initial guesses to try
plot_dir = '/Users/angi/NIS/Analysis/bolotest/plots/layer_extraction_analysis/'
fn_comments = ''   # filename comments

bolos = np.array(['bolo 1b', 'bolo 24', 'bolo 23', 'bolo 22', 'bolo 21', 'bolo 20', 'bolo 7', 'bolo 13'])
# p0 = np.array([0.89, 0.3, 1.3, .58, 1.4, 1.2])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]; Joel's results from chi-sq min sim
p0 = np.array([0.89, 0.3, 1.3, .58, 1, 1])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]; constrained alpha range
ydata = np.array([11.7145073, 4.921841228, 8.077815536, 10.03001622, 16.63099617, 5.386790491, 15.2863792, 3.585251305])   # pW/K at Tc, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13*
sigma = np.array([0.100947739, 0.063601732, 0.078665632, 0.130040288, 0.142600818, 0.059261252, 0.123206779, 0.052084114])   # pW/K at Tc; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13*
# bounds1 = [(0, 0, 0, 0, 0, 0), (np.inf, np.inf, np.inf, 2, 2, 2)]; bounds2 = [(0, np.inf), (0, np.inf), (0, np.inf), (0, 2), (0, 2), (0, 2)]   # different fitting routines like different formats
bounds1 = [(0, 0, 0, 0, 0, 0), (np.inf, np.inf, np.inf, 1, 1, 1)]; bounds2 = [(0, np.inf), (0, np.inf), (0, np.inf), (0, 1), (0, 1), (0, 1)]   # constrained alpha version
ydata_J = np.array([13.74, 4.921841228, 8.077815536, 10.03001622, 16.63099617, 5.386790491, 15.2863792, 3.585251305])   # pW/K at Tc, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13*
sigma_J = np.array([0.17, 0.063601732, 0.078665632, 0.130040288, 0.142600818, 0.059261252, 0.123206779, 0.052084114])   # pW/K at Tc; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13*
# ydata = ydata_J; sigma = sigma_J   # see if QPs match Joel's results
fits_J = np.array([.89, 0.30, 1.3277, 0.58, 1.42, 1.200]); error_J = np.array([0.14, 0.18, 0.02, 0.57, 0.53, 0.06])
# boundsred = [(0, np.inf), (0, 2)]   # G, alpha
boundsred = [(0, np.inf), (0, 1)]   # G, constraiend alpha

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
if run_sim_allthree:   # run simulation for all three minimization routines
    LS_params, LS_std, CS_params, CS_std, func_params, func_std = run_sim(n_its, p0, data, bounds1, bounds2, plot_dir, show_yplots=show_plots, fn_comments=fn_comments, save_figs=save_figs)

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

if reduce_min:   # troubleshoot minimization by reducing number of parameters

    LS_params, LS_std, CS_params, CS_std, func_params, func_std = run_sim(n_its, p0, data, bounds1, bounds2, plot_dir, show_yplots=show_plots, save_figs=save_figs, fn_comments=fn_comments)
    full_res = np.array([func_params, func_std])   # include full MC simulation results on QPs
    U_single, U_MC, Uerr_MC = overplot_qp(p0, data, boundsred, n_its, fits_J, 'U', plot_dir, full_res=full_res, savefigs=save_figs, fn_comments=fn_comments)
    W_single, W_MC, Werr_MC = overplot_qp(p0, data, boundsred, n_its, fits_J, 'W', plot_dir, full_res=full_res, savefigs=save_figs, fn_comments=fn_comments)
    I_single, I_MC, Ierr_MC = overplot_qp(p0, data, boundsred, n_its, fits_J, 'I', plot_dir, full_res=full_res, savefigs=save_figs, fn_comments=fn_comments)

if average_qp:

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
    # funcgrid_W, funcgridmean_W = plot_meanqp(p0, data, n_its, 'W', plot_dir, savefigs=save_figs, fn_comments=fn_comments)
    # funcgrid_I, funcgridmean_I = plot_meanqp(p0, data, n_its, 'I', plot_dir, savefigs=save_figs, fn_comments=fn_comments)



plt.show()