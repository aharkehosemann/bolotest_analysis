import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from scipy.optimize import curve_fit, minimize, minimize_scalar
from bolotest_routines import *
# from scipy.stats import chi2

"""
Script for measuring individual layer contributions to the total thermal conductivity of a TES. 
Data from 'bolotest' measurements run in 2018. 
G_total and error is from TES power law fit measured at Tc and scaled to 170 mK, assuming dP/dT = G = n*k*Tc^(n-1).
Layers: U = SiN + SiO substrate, W = Nb wiring layers (<=2 layers per leg), I = SiN insulating layers (<=2 layers per leg)

aharkehosemann@gmail.com
2022/01

UPDATE 2022.06.15 : We recently discovered the wiring layers have a shorter width (W1=5um, W2=3um) than the SiN layers, and that there's a 120nm SiOx layer on the membrane under the 300nm SiNx layer. 
I now scale W2 terms by 3/5 and W stack by thickness-weighted average width (assumes G scales linearly with layer width). 
Shannon estimates 11-14 nm etching on the SiOx, so true thickness is ~109 nm. 

UPDATE 2022.06.28 : Should we model the data as five layers (treat substrate as 420 nm thick layer of one material) or six layers (one 120 nm SiOx and one 300nm layer SiNx)?
DOF of six-layer fit is 0...
What I've been referring to as "function to minimize" is actually weighted least squares. 
Not sure if looking at significance is useful because we aren't hypothesis testing and have very small DOF.

TODO: phonon heat capacity for Nb shouldn't have NA but N; true error analysis for G predictions from model uncertainty
"""

### User Switches
# which analysis?
runsim = False   # run MC simulation for extracting fit values and variances
random_initguess = False   # try simulation with randomized initial guesses
quality_plots = False   # map G_x vs alpha_x space, plot full sim results + Joel's results + two-param results + single value results
average_qp = False   # show average value for 2D quality plot over MC sim to check for scattering
lit_compare = False
legacy_data = True   # compare with NIST sub-mm bolo legacy data
six_layers = False
design_implications = False

# options
save_figs = False   
save_sim = False   # save full simulation
show_plots = False   # show simulated y-data plots during MC simulation

n_its = int(1E4)   # number of iterations for MC simulation
num_guesses = 100   # number of randomized initial guesses to try
plot_dir = '/Users/angi/NIS/Analysis/bolotest/plots/layer_extraction_analysis/'
fn_comments = '_fullGerroranalysis_alphares'   # filename comments
alim = [0, 1]   # limits for fitting alpha
# p0 = np.array([0.89, 0.3, 1.3, .58, 1, 1])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]; constrained alpha range
# p0 = np.array([0.73, 0.62, 1.30, 0.50, 1.11, 1.27])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]; fit results with new w width and U thickness
p0 = np.array([0.72, 0.60, 1.27, 0.33, 1.02, 1.29]); sigma_p0 = np.array([0.06, 0.12, 0.05, 0.39, 0.80, 0.09])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]; fit results with full G(170mK) error analysis & their errors
p0_res = np.array([0.8, 0.42, 1.33, 1., 1., 1.]); sigmap0_res = np.array([0.03, 0.06, 0.03, 0.03, 0.03, 0.0])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]; fit results with full G(170mK) error analysis & their errors

# choose data set
ydata_Tc = np.array([11.7145073, 4.921841228, 8.077815536, 10.03001622, 16.63099617, 5.386790491, 15.2863792, 3.585251305])   # pW/K at Tc, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13*; bolo1b is weighted average (not sure we trust row 10)
sigma_Tc = np.array([0.100947739, 0.063601732, 0.078665632, 0.130040288, 0.142600818, 0.059261252, 0.123206779, 0.052084114])   # pW/K at Tc; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13*
ydata_J = np.array([13.74, 4.921841228, 8.077815536, 10.03001622, 16.63099617, 5.386790491, 15.2863792, 3.585251305])   # pW/K at Tc, most were weighted averages*; bolo 1b, 24*, 23*, 22, 21*, 20*, 7*, 13*; only row 5 for bolo1b
sigma_J = np.array([0.17, 0.063601732, 0.078665632, 0.130040288, 0.142600818, 0.059261252, 0.123206779, 0.052084114])   # pW/K at Tc; bolo 1b, 24*, 23*, 22, 21*, 20*, 7*, 13*
# ydata_170mK = [13.74098652, 4.91314256, 8.006715762, 10.03001622, 16.49000297, 5.362145301, 15.14067701, 3.5772749]   # pW/K at 170 mK, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13* (only using row 5 value of bolo1b)
# sigma_170mK = [0.171320173, 0.087731049, 0.108034223, 0.130040288, 0.19732244, 0.078047237, 0.166570199, 0.072651667]
ydata_170mK = [13.51632171, 4.889791292, 7.929668225, 9.925580294, 16.27276237, 5.27649525, 14.95079826, 3.577979915]   # pW/K at 170 mK, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13* (only using row 5 value of bolo1b), full G error analysis
sigma_170mK = [0.396542, 0.08774166, 0.148831461, 0.22016008, 0.411908086, 0.079748858, 0.340424395, 0.074313687]
ydata = np.array(ydata_170mK); sigma = np.array(sigma_170mK)

bolos = np.array(['bolo 1b', 'bolo 24', 'bolo 23', 'bolo 22', 'bolo 21', 'bolo 20', 'bolo 7', 'bolo 13'])
bounds1 = [(0, 0, 0, alim[0], alim[0], alim[0]), (np.inf, np.inf, np.inf, alim[1], alim[1], alim[1])]; bounds2 = [(0, np.inf), (0, np.inf), (0, np.inf), (alim[0], alim[1]), (alim[0], alim[1]), (alim[0], alim[1])]   # different fitting routines like different formats
fits_J = np.array([.89, 0.30, 1.3277, 0.58, 1.42, 1.200]); error_J = np.array([0.14, 0.18, 0.02, 0.57, 0.53, 0.06])   # compare with Joel's results
results_J = [fits_J, error_J]   # Joel's fit parameters and error
boundsred = [(0, np.inf), (alim[0], alim[1])]   # G, alpha
sim_file = plot_dir + 'sim' + fn_comments + '.pkl'

### Geometries; these are no longer the true thicknesses
# coeffs = np.array([[[4, 0, 0], [4, 4, 0], [4, 4, 0]],   # bolo 1b
#         [[1, 3, 0], [1, 1, 0], [1, 1, 0]],   # bolo 24
#         [[2, 2, 0], [2, 2, 0], [2, 2, 0]],   # bolo 23
#         [[3, 1, 0], [3, 3, 0], [3, 3, 0]],   # bolo 22
#         [[3, 1, 0], [1, 1, 0], [1, 3, 0]],   # bolo 21
#         [[4, 0, 0], [3, 1, 1], [1, 0, 0]],   # bolo 20
#         [[3, 1, 0], [3, 3, 0], [3, 1, 3]],   # bolo 7
#         [[3, 1, 0], [1, 1, 0], [1, 0, 0]]])   # bolo 13

# ratios = np.array([[[1, 0, 0], [160/400, 340/400, 0], [350/400, 1, 0]],   # bolo 1b
#         [[1, 220/300, 0], [160/400, 340/400, 0], [350/400, 1, 0]],   # bolo 24
#         [[1, 220/300, 0], [160/400, 340/400, 0], [350/400, 1, 0]],   # bolo 23
#         [[1, 220/300, 0], [160/400, 340/400, 0], [350/400, 1, 0]],   # bolo 22
#         [[280/300, 1, 0], [160/400, 285/400, 0], [350/400, (270+400)/400, 0]],   # bolo 21
#         [[1, 0, 0], [(100+285)/400, 160/400, 285/400], [350/400, 0, 0]],   # bolo 20
#         [[1, 280/300, 0], [160/400, 340/400, 0], [350/400, (270+400)/400, 1]],   # bolo 7
#         [[220/300, 1, 0], [160/400, 285/400, 0], [350/400, 0, 0]]])   # bolo 13

# xdata = np.array([coeffs, ratios])
data = [ydata, sigma] 

### Execute Analysis
if runsim:   # run simulation with just hand-written function
    func_params, func_std = runsim_WLS(n_its, p0, data, bounds2, plot_dir, show_yplots=show_plots, save_figs=save_figs, fn_comments=fn_comments, sim_file=sim_file, save_sim=save_sim, calc_Gwire=True, model='default')


if quality_plots:   # plot G_x vs alpha_x parameter space with various fit results

    L = 220   # TES leg length, um
    full_res = runsim_WLS(n_its, p0, data, bounds2, plot_dir, show_yplots=show_plots, save_figs=save_figs, fn_comments=fn_comments, calc_Gwire=True)  
    params_fit, sigmas_fit =  full_res
    # dof_five = len(ydata) - len(p0) - 1   # degrees of freedom for five-layer fit; number of data points - number of parameters - 1
    U_MC, Uerr_MC = overplot_qp(p0, data, boundsred, n_its, results_J, 'U', plot_dir, full_res=full_res, savefigs=save_figs, fn_comments=fn_comments, vmax=500)
    W_MC, Werr_MC = overplot_qp(p0, data, boundsred, n_its, results_J, 'W', plot_dir, full_res=full_res, savefigs=save_figs, fn_comments=fn_comments, vmax=500)
    I_MC, Ierr_MC = overplot_qp(p0, data, boundsred, n_its, results_J, 'I', plot_dir, full_res=full_res, savefigs=save_figs, fn_comments=fn_comments, vmax=500)

    Gmeas_U, Gmeas_W, Gmeas_I, alpham_U, alpham_W, alpham_I = params_fit; sigGU, sigGW, sigGI, sigalpha, sigalphaW, sigalphaI = sigmas_fit
    kappam_U = GtoKappa(Gmeas_U, 7*300E-3, L); sigkappam_U = GtoKappa(sigGU, 7*300E-3, L)   # pW / K / um
    kappam_W = GtoKappa(Gmeas_W, 5*400E-3, L); sigkappam_W = GtoKappa(sigGW, 5*400E-3, L)   # pW / K / um
    kappam_I = GtoKappa(Gmeas_I, 7*400E-3, L); sigkappam_I = GtoKappa(sigGI, 7*400E-3, L)   # pW / K / um
    print('Kappa_U: ', round(kappam_U, 2), ' +/- ', round(sigkappam_U, 2), ' pW/K um')
    print('Kappa_W: ', round(kappam_W, 2), ' +/- ', round(sigkappam_W, 2), ' pW/K um')
    print('Kappa_I: ', round(kappam_I, 2), ' +/- ', round(sigkappam_I, 2), ' pW/K um')

    WLS_fit = WLS_val(params_fit, data)
    print('WLS value for the five-layer fit: ', round(WLS_fit, 3))   # the function we've been minimizing is not actually the chi-squared value...

    chisq_fit = calc_chisq(ydata, Gbolos(params_fit))
    print('Chi-squared value for the five-layer fit: ', round(chisq_fit, 3))


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
            data_it = [y_its[ii], sigma]   
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
    # molvol = 1.083E-5   # m^3 per mol
    L = 220   # TES leg length, um

    def phonon_wlength(vs, T):   # returns dominant phonon wavelength in nm
        # vs for SiN is 6986 m/s
        # vs for Nb is 3480 m/s
        return hbar*np.pi*vs/(kB*T)*1E9

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

    # def GtoKappa(G, A, L):   # thermal conductance, area, length
    #     return G*L/A

    Gmeas_U = 0.725E-12; sigGU = 0.07E-12   # W / K, results from SiN + SiO combined substrate layer
    Gmeas_W = 0.69E-12; sigGW = 0.14E-12   # our measured thermal conductance of 400nm film of Nb, width = 5um, W / K
    Gmeas_I = 1.32E-12; sigGI = 0.03E-12   # W / K
    alpham_U = 0.17; sigGU = 0.49   # W / K
    alpham_W = 1.68; sigGW = 0.56   # our measured thermal conductance of 400nm film of Nb, width = 5um, W / K
    alpham_I = 1.24; sigGI = 0.05   # W / K
    G_stack = Gmeas_W*(2/4)**(alpham_W+1) + Gmeas_I*(350/400)**alpham_I + 3/5*Gmeas_W + Gmeas_I

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



if legacy_data:

    L = 220   # TES leg length, um

    # compare with NIST legacy data
    boloGs = ydata_170mK; sigma_boloGs = sigma_170mK   # choose bolotest G's to compare
    wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20
    ATESs = np.array([(7*4*.420+5*4*.160+3*4*.340+7*4*.350+7*4*.400), (7*1*.420+7*3*.340+5*.160+3*.340+7*.350+7*.400), (7*2*.420+7*2*.340+5*2*.160+3*2*.340+7*2*.350+7*2*.400), (7*3*.420+7*1*.340+5*3*.160+3*3*.340+7*3*.350+7*3*.400), (7*1*.420+7*3*.400+5*1*.160+3*1*.285+7*3*.370+7*1*.350), (7*4*.420+5*1*.160+wstack_width*3*.385+3*1*.285+7*1*.340), (7*3*.420+7*1*.400+5*3*.160+3*1*3.340+7*3*.350+7*1*.670+7*3*.400), (7*1*.420+7*3*.400+5*1*.160+3*1*.285+7*1*.350) ])
    ASiN = np.array([(7*4*.300), (7*1*.300+7*3*.220), (7*2*.300+7*2*.220), (7*3*.300+7*1*.220), (7*1*.300+7*3*.280), (7*4*.300), (7*3*.300+7*1*.280), (7*1*.300+7*3*.280)])   # area of 4 nitride beams
    AoLTESs = ASiN/L   # A/L calculated using just area of nitride to match Shannon's numbers
    kappaTES = GtoKappa(boloGs, ATESs, L*np.ones(len(ATESs))); sigma_kappaTES = GtoKappa(sigma_boloGs, ATESs, L*np.ones(len(ATESs)))   # all 4 legs, A is all layers
    kappaSiN = GtoKappa(boloGs, ASiN, L*np.ones(len(ATESs))); sigma_kappaSiN = GtoKappa(sigma_boloGs, ASiN, L*np.ones(len(ATESs)))   # all 4 legs, A is SiN beam

    dW1 = .160; dI1 = .350; dW2 = .340; dI2 = .400   # film thicknesses, um
    dmicro = dW1 + dI1 + dW2 + dI2
    legacy_Gs = np.array([1296.659705, 276.1, 229.3, 88.3, 44, 76.5, 22.6, 644, 676, 550, 125, 103, 583, 603, 498, 328, 84, 77, 19, 12.2, 10.5, 11.7, 13.1, 9.98, 16.4, 8.766, 9.18, 8.29, 9.57, 7.14, 81.73229733, 103.2593154, 106.535245, 96.57474779, 90.04141806, 108.616653, 116.2369491, 136.2558345, 128.6066776, 180.7454359, 172.273248, 172.4456603, 192.5852409, 12.8, 623, 600, 620, 547, 636, 600.3, 645, 568, 538.7, 491.3, 623, 541.2, 661.4, 563.3, 377.3, 597.4, 395.3, 415.3, 575, 544.8, 237.8, 331.3, 193.25, 331.8, 335.613, 512.562, 513.889, 316.88, 319.756, 484.476, 478.2, 118.818, 117.644, 210.535, 136.383, 130.912, 229.002, 236.02, 101.9, 129.387, 230.783, 230.917, 130.829, 127.191, 232.006, 231.056])  
    legacy_ll = np.array([250, 61, 61, 219.8, 500, 500, 1000, 50, 50, 50, 100, 300, 50, 50, 50, 100, 100, 300, 500, 1000, 1000, 1000, 1000, 1250, 500, 1000, 1000, 1000, 1000, 1250, 640, 510, 510, 510, 510, 510, 730, 610, 490, 370, 370, 370, 300, 500, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
    legacy_lw = np.array([25, 14.4, 12.1, 10, 10, 15, 10, 41.5, 41.5, 34.5, 13, 16.5, 41.5, 41.5, 34.5, 29, 13, 16.5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 7, 7, 7, 7, 7, 7, 10, 10, 8, 8, 8, 8, 8, 6, 41.5, 41.5, 41.5, 41.5, 41.5, 41.5, 41.5, 41.5, 34.5, 34.5, 41.5, 41.5, 34.5, 34.5, 23.6, 37.5, 23.6, 23.6, 37.5, 37.5, 13.5, 21.6, 13.5, 21.6, 23.6, 37.5, 37.5, 23.6, 23.6, 37.5, 37.5, 11.3, 11.3, 18.5, 11.3, 11.3, 18.5, 18.5, 11.3, 11.3, 18.5, 18.5, 11.3, 11.3, 18.5, 18.5])
    legacy_dsub = np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    legacy_Tcs = np.array([557, 178.9, 178.5, 173.4, 170.5, 172.9, 164.7, 163, 162, 163, 164, 164, 168, 168, 167, 167, 165, 166, 156, 158, 146, 146, 149, 144, 155, 158, 146, 141, 147, 141, 485.4587986, 481.037173, 484.9293596, 478.3771521, 475.3010335, 483.4209782, 484.0258522, 477.436482, 483.5417917, 485.8804622, 479.8911157, 487.785816, 481.0323883, 262, 193, 188, 188.8, 188.2, 190.2, 188.1, 186.5, 184.5, 187.5, 185.5, 185.8, 185.6, 185.7, 183.3, 167.3, 167, 172.9, 172.8, 166.61, 162.33, 172.87, 161.65, 163.06, 166.44, 177.920926, 178.955154, 178.839062, 177.514658, 177.126927, 178.196297, 177.53632, 169.704602, 169.641018, 173.026393, 177.895192, 177.966456, 178.934122, 180.143125, 177.16833, 178.328865, 179.420334, 179.696264, 172.724501, 172.479515, 177.385267, 177.492689])
    legacy_ns = np.array([3.5, 3.4, 3.4, 3, 2.8, 3, 2.7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2.707252717, 2.742876666, 2.741499631, 2.783995279, 2.75259088, 2.796872814, 2.747211811, 2.782265754, 2.804876038, 2.879595447, 2.871133545, 2.889243695, 2.870571891, 2.6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    legacy_Gs170 = scale_G(.170, legacy_Gs, legacy_Tcs/1E3, legacy_ns)   # scale G's to Tc of 170 mK
    legacy_A = 4*(legacy_dsub + dmicro)*legacy_lw   # um^2, area for four legs, thickness is substrate + wiring stack
    legacy_AoLs = legacy_A/legacy_ll  # um
    legacy_kappas = GtoKappa(legacy_Gs, legacy_A, legacy_ll)   # kappa at bolo Tc
    legacy_kappas170 = GtoKappa(legacy_Gs170, legacy_A, legacy_ll)   # kappa at 170 mK    
    # inds160 = np.array([*np.arange(4,33), *np.arange(46,93)])-3   # Tc~160 mK bolometers from Shannon's data
    # inds450 = np.array([3, *np.arange(33,46)])-3   # Tc~450 mK bolometers from Shannon's data
    # lbind = 44   # lightbird index, only bolo with Tc<400 mK that has dsub=1um
    lTcinds = np.where(legacy_Tcs<200)[0]   # Tc<200 mK bolometers from Shannon's data
    hTcinds = np.where(legacy_Tcs>=200)[0]   # Tc<200 mK bolometers from Shannon's data

    plt.figure()   # compare pseudo kappas
    plt.scatter(legacy_AoLs[lTcinds], legacy_Gs[lTcinds], color='b', alpha=.5, label="Legacy Data, low Tc")
    plt.scatter(legacy_AoLs[hTcinds], legacy_Gs[hTcinds], color='r', alpha=.5, label="Legacy Data, high Tc")
    plt.ylabel('TES G(Tc) [pW/K]')
    plt.xlabel('Leg A/L [um]')
    plt.legend()
    plt.yscale('log'); plt.xscale('log')
    plt.title('NIST TES Legacy G Data')
    if save_figs: plt.savefig(plot_dir + 'legacydata_Gs_loglog.png', dpi=300) 


    plt.figure()   # compare pseudo kappas
    # plt.scatter(legacy_AoLs[lTcinds], legacy_kappas[lTcinds], color='b', alpha=.5, label="Legacy Data, 160 mK")
    plt.scatter(legacy_AoLs[lTcinds], legacy_kappas170[lTcinds], color='g', alpha=.5, label="Legacy Data, 170 mK")
    # plt.scatter(legacy_AoLs[ hTcinds], legacy_kappas[ hTcinds], color='r', alpha=.5, label="Legacy Data, 450 mK")
    plt.errorbar(AoLTESs, kappaTES, yerr=sigma_kappaTES, marker="^", color='purple', label='Bolotest', ls='none')
    for bb, bolo in enumerate(bolos):
        # plt.errorbar(AoLTESs[bb], kappaTES[bb], yerr=sigma_kappaTES[bb], marker="^", color='p')
        plt.annotate(bolo.split(' ')[1], (AoLTESs[bb]+0.02, kappaTES[bb]-2), color='purple')
    plt.ylabel('TES $\kappa$ [pW/K/um]')
    plt.xlabel('Leg A/L [um]')
    plt.legend()
    plt.title('Pseudo $\kappa$ = G_TES * L_leg / A_legs')
    if save_figs: plt.savefig(plot_dir + 'legacydata_pseudokappa.png', dpi=300) 


    legacy_Gpredkappa = Gfromkappas(legacy_dsub, legacy_lw, legacy_ll)
    legacy_GpredkappaW = Gfromkappas(legacy_dsub, legacy_lw, legacy_ll, layer='wiring')
    legacy_GpredkappaU = Gfromkappas(legacy_dsub, legacy_lw, legacy_ll, layer='U')
    predkappa_bolo1b = Gfromkappas(.420, 7, 220)
    predkappa_bolo1bU = Gfromkappas(.420, 7, 220, layer='U')
    predkappa_bolo1bW = Gfromkappas(.420, 7, 220, layer='wiring')

    ### predictions from kappa
    plt.figure()   # plot predicted G's from kappas and measured G's
    plt.scatter(legacy_AoLs[lTcinds], legacy_Gs170[lTcinds], color='g', alpha=.5, label="Legacy Data, 170 mK")
    plt.scatter(legacy_AoLs[lTcinds], legacy_Gpredkappa[lTcinds], color='k', alpha=.5, marker='*', label="Prediction, Total")
    plt.scatter(legacy_AoLs[lTcinds], legacy_GpredkappaU[lTcinds], color='blue', alpha=.5, marker='+', label="Prediction, Substrate")
    plt.scatter(legacy_AoLs[lTcinds], legacy_GpredkappaW[lTcinds], color='mediumpurple', alpha=.5, marker='x', s=20, label="Prediction, Microstrip")
    plt.errorbar(AoLTESs[0], boloGs[0], yerr=sigma_boloGs[0], marker='o', markersize=5, color='purple', label='bolo 1b')
    plt.scatter(AoLTESs[0], predkappa_bolo1b, color='purple', marker='*', label="Prediction, bolo1b")
    plt.scatter(AoLTESs[0], predkappa_bolo1bU, color='purple', marker='+')
    plt.scatter(AoLTESs[0], predkappa_bolo1bW, color='purple', marker='x', s=20)
    plt.legend()
    plt.ylabel('G(170 mK) [pW/K]')
    plt.xlabel('Leg A/L [um]')
    plt.title("G Predictions from Layer $\kappa$'s")
    plt.yscale('log'); plt.xscale('log')
    if save_figs: plt.savefig(plot_dir + 'legacydata_Gpredfromkappa_loglog.png', dpi=300) 


    plt.figure()   # residuals from kappa predictions
    plt.axhline(0, color='k', alpha=0.6)
    plt.scatter(legacy_AoLs[lTcinds], legacy_Gs170[lTcinds] - legacy_Gpredkappa[lTcinds], color='r')
    plt.ylabel('Residuals [pW/K]')
    plt.xlabel('Leg A/L [um]')
    plt.title("Measured G - Predicted G from Layer $\kappa$'s")
    if save_figs: plt.savefig(plot_dir + 'legacydata_Gpredfromkappa_residuals.png', dpi=300) 


    ### predictions from model
    legacy_Gpredmodel = Gfrommodel(p0, legacy_dsub, legacy_lw, legacy_ll)
    # sigma_Gmodel = Gfrommodel(sigma_p0, legacy_dsub, legacy_lw, legacy_ll)   # sigma_G from model uncertainty
    legacy_GpredmodelW = Gfrommodel(p0, legacy_dsub, legacy_lw, legacy_ll, layer='wiring')
    legacy_GpredmodelU = Gfrommodel(p0, legacy_dsub, legacy_lw, legacy_ll, layer='U')
    predmodel_1b = Gfrommodel(p0, .420, 7, 220)
    predmodel_1bW = Gfrommodel(p0, .420, 7, 220, layer='wiring')
    predmodel_1bU = Gfrommodel(p0, .420, 7, 220, layer='U')
    # stres_model = (legacy_Gs170 - legacy_Gpredmodel)/sigma_Gmodel   # standardized residuals
    normres_model = (legacy_Gs170 - legacy_Gpredmodel)/legacy_Gs170   # % residuals

    plt.figure()   # plot predicted G's from model and measured G's
    plt.scatter(legacy_AoLs[lTcinds], legacy_Gs170[lTcinds], color='g', alpha=.5, label="Legacy Data, 170 mK")
    plt.scatter(legacy_AoLs[lTcinds], legacy_Gpredmodel[lTcinds], color='k', alpha=.5, marker='*', label="Model Pred, Total")
    # plt.errorbar(legacy_AoLs[lTcinds], legacy_Gpredmodel[lTcinds], yerr=sigma_Gmodel[lTcinds], color='k', alpha=.5, marker='*', linestyle='None', label="Model Pred, Total")
    plt.scatter(legacy_AoLs[lTcinds], legacy_GpredmodelU[lTcinds], color='blue', alpha=.5, marker='+', label="Model Pred, Substrate")
    plt.scatter(legacy_AoLs[lTcinds], legacy_GpredmodelW[lTcinds], color='mediumpurple', alpha=.5, marker='x', s=20, label="Model Pred, Microstrip")    
    # plt.scatter(legacy_AoLs[lTcinds], legacy_GpredmodelU[lTcinds], color='k', alpha=.5, marker='+', label="Model Pred, Substrate")
    # plt.scatter(legacy_AoLs[lTcinds], legacy_GpredmodelW[lTcinds], color='k', alpha=.5, marker='x', s=20, label="Model Pred, Microstrip")
    plt.errorbar(AoLTESs[0], boloGs[0], yerr=sigma_boloGs[0], marker='o', markersize=5, color='purple', label='bolo 1b')
    plt.scatter(AoLTESs[0], predmodel_1b, marker='*', color='purple', label="Model Pred, bolo1b")
    plt.scatter(AoLTESs[0], predmodel_1bU, marker='+', s=20, color='purple')
    plt.scatter(AoLTESs[0], predmodel_1bW, marker='x', color='purple')
    plt.legend()
    plt.ylabel('G(170 mK) [pW/K]')
    plt.xlabel('Leg A/L [um]')
    plt.title("G Predictions from Model")
    plt.yscale('log'); plt.xscale('log')
    if save_figs: plt.savefig(plot_dir + 'legacydata_Gpredfrommodel_loglog.png', dpi=300) 

    plt.figure()   # residuals from model predictions
    plt.axhline(0, color='k', alpha=0.6)
    plt.scatter(legacy_AoLs[lTcinds], normres_model[lTcinds], color='r')
    plt.ylabel('Residuals [% measured G]')
    plt.xlabel('Leg A/L [um]')
    plt.title("Normalized Residuals")
    if save_figs: plt.savefig(plot_dir + 'legacydata_Gpredfrommodel_normresiduals.png', dpi=300) 

    ### predictions from model, restricted alpha
    legacy_Gpredmodel_res = Gfrommodel(p0_res, legacy_dsub, legacy_lw, legacy_ll)
    legacy_GpredmodelW_res = Gfrommodel(p0_res, legacy_dsub, legacy_lw, legacy_ll, layer='wiring')
    legacy_GpredmodelU_res = Gfrommodel(p0_res, legacy_dsub, legacy_lw, legacy_ll, layer='U')
    predmodel_1b_res = Gfrommodel(p0_res, .420, 7, 220)
    predmodel_1bW_res = Gfrommodel(p0_res, .420, 7, 220, layer='wiring')
    predmodel_1bU_res = Gfrommodel(p0_res, .420, 7, 220, layer='U')
    normres_resmodel = (legacy_Gs170 - legacy_Gpredmodel_res)/legacy_Gs170   # % residuals

    plt.figure()   # plot predicted G's from model and measured G's -- low Tc and high Tc separated
    plt.scatter(legacy_AoLs[lTcinds], legacy_Gs170[lTcinds], color='b', alpha=.5, label="Low Tc Legacy Data, scaled to 170 mK")
    plt.scatter(legacy_AoLs[hTcinds], legacy_Gs170[hTcinds], color='r', alpha=.5, label="High Tc Legacy Data, scaled to 170 mK")
    plt.scatter(legacy_AoLs[lTcinds], legacy_Gpredmodel[lTcinds], color='b', alpha=.5, marker='*')
    plt.scatter(legacy_AoLs[hTcinds], legacy_Gpredmodel[hTcinds], color='r', alpha=.5, marker='*', label="High Tc Model Pred, Total")
    plt.scatter(legacy_AoLs[lTcinds], legacy_GpredmodelU[lTcinds], color='b', alpha=.5, marker='+')
    plt.scatter(legacy_AoLs[hTcinds], legacy_GpredmodelU[hTcinds], color='r', alpha=.5, marker='+', label="High Tc Model Pred, Substrate")
    # plt.errorbar(AoLTESs[0], boloGs[0], yerr=sigma_boloGs[0], marker='o', markersize=5, color='purple', label='bolo 1b')
    # plt.scatter(AoLTESs[0], predmodel_bolo1b, marker='x', color='purple', label="Model Pred, bolo1b")
    plt.legend()
    plt.ylabel('G(170 mK) [pW/K]')
    plt.xlabel('Leg A/L [um]')
    plt.title("G Predictions from Model")
    plt.yscale('log'); plt.xscale('log')
    if save_figs: plt.savefig(plot_dir + 'legacydata_Gpredfrommodel_lowandhighTc_loglog.png', dpi=300) 

    plt.figure(figsize=(7,6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    ax1 = plt.subplot(gs[0])   # model vs data
    plt.scatter(legacy_AoLs[lTcinds], legacy_Gs170[lTcinds], color='g', alpha=.5, label="Legacy Data, 170 mK")
    plt.scatter(legacy_AoLs[lTcinds], legacy_Gpredmodel_res[lTcinds], color='k', alpha=.5, marker='*', label="Model Pred, Total")
    plt.scatter(legacy_AoLs[lTcinds], legacy_GpredmodelU_res[lTcinds], color='blue', alpha=.5, marker='+', label="Model Pred, Substrate")
    plt.scatter(legacy_AoLs[lTcinds], legacy_GpredmodelW_res[lTcinds], color='mediumpurple', alpha=.5, marker='x', s=20, label="Model Pred, Microstrip")    
    plt.errorbar(AoLTESs[0], boloGs[0], yerr=sigma_boloGs[0], marker='o', markersize=5, color='purple', label='bolo 1b')
    plt.scatter(AoLTESs[0], predmodel_1b_res, marker='*', color='purple', label="Model Pred, bolo1b")
    plt.scatter(AoLTESs[0], predmodel_1bU_res, marker='+', color='purple')
    plt.scatter(AoLTESs[0], predmodel_1bW_res, marker='x', s=20, color='purple')
    plt.legend()
    plt.ylabel('G(170 mK) [pW/K]')
    plt.title("G Predictions from Model, Restricted $\\alpha}$")
    plt.yscale('log'); plt.xscale('log')

    ax2 = plt.subplot(gs[1], sharex=ax1)   # model residuals
    plt.axhline(0, color='k', ls='--')
    plt.scatter(legacy_AoLs[lTcinds], normres_resmodel[lTcinds], color='r')
    plt.ylabel("Res'ls [% G_meas]")
    plt.xlabel('Leg A/L [um]')
    plt.subplots_adjust(hspace=.0)   # merge to share one x axis
    if save_figs: plt.savefig(plot_dir + 'Gpredfrommodel_resalpha.png', dpi=300) 

    # looking for crossover dsub value where G_micro = G_sub 
    # dsubs = np.linspace(0,1,int(1E3))
    # G_micros = Gfrommodel(p0, dsubs, 20, 10, layer='wiring')
    # G_Us = Gfrommodel(p0, dsubs, 20, 10, layer='U')
    # Gratio = G_Us/G_micros
    # xover = np.where(Gratio>=1)[0][0]



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
    G_stack = Gmeas_W6*(2/4)**(alpham_W6+1) + Gmeas_I6*(350/400)**(alpham_I6+1) + 3/5*Gmeas_W6 + Gmeas_I6
    print('G of full wire stack:', round(G_stack, 2), ' pW / K')   # W1 + W2 + I1 + I2 as deposted; pW / K

    kappam_SiO6 = GtoKappa(Gmeas_SiO6, 7*109E-3, L); sigkappam_SiO6 = GtoKappa(sigGSiO6, 7*109E-3, L)   # pW / K / um
    kappam_SiN6 = GtoKappa(Gmeas_SiN6, 7*300E-3, L); sigkappam_SiN6 = GtoKappa(sigGSiN6, 7*300E-3, L)   # pW / K / um
    kappam_W6 = GtoKappa(Gmeas_W6, 5*400E-3, L); sigkappam_W6 = GtoKappa(sigGW6, 5*400E-3, L)   # pW / K / um
    kappam_I6 = GtoKappa(Gmeas_I6, 7*400E-3, L); sigkappam_I6 = GtoKappa(sigGI6, 7*400E-3, L)   # pW / K / um
    print('Kappa_SiO: ', round(kappam_SiO6, 2), ' +/- ', round(sigkappam_SiO6, 2), ' pW/K um')
    print('Kappa_SiN: ', round(kappam_SiN6, 2), ' +/- ', round(sigkappam_SiN6, 2), ' pW/K um')
    print('Kappa_W: ', round(kappam_W6, 2), ' +/- ', round(sigkappam_W6, 2), ' pW/K um')
    print('Kappa_I: ', round(kappam_I6, 2), ' +/- ', round(sigkappam_I6, 2), ' pW/K um')

    WLS_six = functomin_sixlayers(params_six, data)
    print('WLS Value for the six-layer fit: ', round(WLS_six, 3))

    chisq_six = calc_chisq_test(ydata, Gbolos_six(params_six, data))
    print('Chi-squared for the six-layer fit: ', round(chisq_six, 3))


if design_implications:   # making plots to illustrate TES and NIS design implications of this work

    def GTES(params, Swidth, wiring='full'):  # width in microns
        # calculate total G of a TES with two bare substrate legs and two legs with a full wiring stack, same geometry as our bolo 23
        
        GU, GW, GI, aU, aW, aI = params
        stack_width = 7.   # wire stack width, um
        W_width = 5.   # W layer width, um
        min_sw = 3.   # minimum wire stack width, um
        min_ww = 1.   # minimum wire layer width, um

        if wiring=='full':   # full stack on two legs
            Gwire = G_wirestack(params)*np.ones_like(Swidth)
            naninds = np.where(Swidth<=min_sw)[0]   # wiring stack can not be smaller than 3um, return nans
            scaleinds = np.where((min_sw<Swidth)&(Swidth<=stack_width))[0]   # wiring layer cannot be widing than substrate, scale by width
            Gwire[naninds] = np.nan
            Gwire[scaleinds] = G_wirestack(params)*Swidth[scaleinds]/7
            return 2*GU*(1 + ((220+120)/420)**(aU+1))*Swidth/7 + 2*Gwire
        if wiring=='W1':   # one 200 nm thick by 5 um wide layer of Nb
            Gwire = GW*(200/400)**(aW+1)*np.ones_like(Swidth)
            naninds = np.where(Swidth<=min_ww)[0]   # wiring layer can not be smaller than 1um, return nans
            scaleinds = np.where((min_ww<Swidth)&(Swidth<=W_width))[0]   # wiring layer cannot be widing than substrate, scale by width
            Gwire[naninds] = np.nan
            Gwire[scaleinds] = GW*(200/400)**(aW+1)*Swidth[scaleinds]/5
            return 2*GU*(1 + ((220+120)/420)**(aU+1))*Swidth/7 + 2*Gwire
        if wiring=='none':   # bare substrate
            return 4*GU*((220+120)/420)**(aU+1)*Swidth/7
        else:
            print('Wiring options are "full" "W1" and "none".')
            return         

    ### plot G_TES and TFN as a function of substrate width
    # Iwidths = np.linspace(3,7)
    Swidths = np.linspace(0.1,100/7)
    Tbath = 0.170   # K
    GTESs_full = GTES(p0, Swidths, wiring='full')   # pW / K (if fitted G's are in pW / K), full wiring stack
    GTESs_W1 = GTES(p0, Swidths, wiring='W1')   # just W1 on substrate
    GTESs_bare = GTES(p0, Swidths, wiring='none')   # bare substrate
    NEPs_full = TFNEP(Tbath, GTESs_full*1E-12)*1E18   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK
    NEPs_W1 = TFNEP(Tbath, GTESs_W1*1E-12)*1E18   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK
    NEPs_bare = TFNEP(Tbath, GTESs_bare*1E-12)*1E18   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK
    # GTES(p0, 7)   # pW / K, this should be the same result as bolo 23
    # Glims = np.array([np.min([GTESs_full, GTESs_W1, GTESs_bare])-0.01*np.max([GTESs_full, GTESs_W1, GTESs_bare]), np.max([GTESs_full, GTESs_W1, GTESs_bare])+0.01*np.max([GTESs_full, GTESs_W1, GTESs_bare])])
    Glims = np.array([0, 1.4*np.nanmax([GTESs_full, GTESs_W1, GTESs_bare])])
    NEPlims = TFNEP(Tbath, Glims*1E-12)*1E18
    
    # predicted G and NEP vs substrate width
    fig, ax1 = plt.subplots() 
    ax1.plot(Swidths, GTESs_full, color='mediumpurple', label='G$_{TES}$, microstrip') 
    ax1.plot(Swidths, GTESs_W1, color='limegreen', label='G$_{TES}$, 200nm Nb') 
    ax1.plot(Swidths, GTESs_bare, color='cornflowerblue', label='G$_{TES}$, Substrate Only') 
    ax1.set_xlabel('Substrate Width [$\mu$m]') 
    ax1.set_ylabel('Predicted G$_{TES}$ [pW/K]') 
    ax1.set_ylim(ymin=Glims[0], ymax=Glims[1]) 
    ax2 = ax1.twinx() 
    ax2.plot(Swidths, NEPs_full, '--', color='mediumpurple', label='NEP, microstrip')   # this varies as G^1/2
    ax2.plot(Swidths, NEPs_W1, '--', color='limegreen', label='NEP, 200nm Nb')   # this varies as G^1/2
    ax2.plot(Swidths, NEPs_bare, '--', color='cornflowerblue', label='NEP, Substrate Only')   # this varies as G^1/2
    ax2.set_ylim(ymin=NEPlims[0], ymax=NEPlims[1]) 
    ax2.set_ylabel('Thermal Fluctuation NEP [aW/$\sqrt{Hz}$]')     
    ax2.set_xlim(0, np.max(Swidths)+np.min(Swidths))
    # ax1.errorbar(7, ydata[2], yerr=sigma[2], marker='.', color='red', markersize='5', linestyle='None', capsize=5, label='Our G Msmt')   # G(bolo 23)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # fig.legend(loc=(0.15,0.7))

    if save_figs: plt.savefig(plot_dir + 'design_implications.png', dpi=300) 


plt.show()