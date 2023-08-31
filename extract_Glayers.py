"""
Script for measuring individual layer contributions to the total thermal conductivity of a TES. 
Data from 'bolotest' measurements run in 2018. 
G_total and error is from TES power law fit measured at Tc and scaled to 170 mK, assuming dP/dT = G = n*k*Tc^(n-1).
Layers: U = SiN + SiO substrate, W = Nb wiring layers (<=2 layers per leg), I = SiN insulating layers (<=2 layers per leg)

aharkehosemann@gmail.com
2022/01


UPDATES
2023/07/12: quote mean values, note bimodality and put total population in SI but note two populations share the same chi-squared

2023/08/09: Added non-equilibrium correction to TF NEP calculation

2023/08/11: There was probably a bookkeeping error during bolotest measurements that swapped Pads 10 and 11 = bolo1c and bolo1b measurements with more BLING. 

TODO: redo six-layer fit to compare chi-squared values?; change kappa dependence on width for W2 layer?
"""
from bolotest_routines import *
from scipy.optimize import fsolve
import csv 


### User Switches
# choose analysis
run_sim = False   # run MC simulation for fitting model
quality_plots = True   # results on G_x vs alpha_x parameter space for each layer
random_initguess = False   # try simulation with randomized initial guesses
average_qp = False   # show average value for 2D quality plot over MC sim to check for scattering
lit_compare = False   # compare measured conductivities with literature values
compare_legacy = False   # compare with NIST sub-mm bolo legacy data
design_implications = False
load_and_plot = False   # scrap; currently replotting separate quality plots into a 1x3 subplot figure
bimodal_solns = False   # compare results of two minima in parameter space found in unconstrained model
compare_modelanddata = True   # plot model predictions and bolotest data
analyze_vlengthdata = False
byeye = False   # pick parameters manually and compare fits
scrap = False

# options
save_sim = True   # save full simulation
save_figs = True   
save_csv = True   # save csv file of resulting parameters
show_plots = False   # show simulated y-data plots during MC simulation
calc_Gwire = False   # calculate G of the wiring stack if it wasn't saved during the simulation
latex_fonts = True
constrained = False   # use constrained model results

n_its = int(1E3)   # number of iterations for MC simulation
# num_guesses = 100   # number of randomized initial guesses to try
analysis_dir = '/Users/angi/NIS/Bolotest_Analysis/'
# fn_comments = '_alpha0inf_1E4iteratinos_fitGconstantTTES_nobling_constrained'; alim = [0,1]   # fitting in [0,1]
# fn_comments = '_alpha0inf_1E4iteratinos_fitGconstantTTES_nobling'; alim = [0, np.inf]   # fitting in [0,infinity)
# fn_comments = '_alphan1top1'; alim = [-1, 1]   # fitting alpha in [-1,1]
# fn_comments = '_alphaninfpinf'; alim = [-np.inf, np.inf]   # fitting alpha in (-infinity,infinity)
# fn_comments = '_alpha0inf_lessbling_varythickness'; alim = [0, np.inf]   # fitting where film thicknesses are allowed to vary
# fn_comments = '_alphaninfpinf_varythickness'; alim = [-np.inf, np.inf]   # fitting manually changing film thicknesses
# fn_comments = '_alphaninfpinf_Gerr5percent'; alim = [-np.inf, np.inf]   # fitting where sigmaG is a % of the value
# fn_comments = '_alphaninfpinf_Gerr3p5percent'; alim = [-np.inf, np.inf]   # fitting where sigmaG is a % of the value
# fn_comments = '_alphan1p1_Gerr3p5percent'; alim = [-1, 1]; sigmaG_frac = 0.035   # fitting where sigmaG is a % of the value
# fn_comments = '_alphan2p2_Gerr3p33percent'; alim = [-2, 2]; sigmaG_frac = 0.0333   # fitting where sigmaG is a % of the value
# fn_comments = '_alphaninfpinf_Gerr3p33percent'; alim = [-np.inf, np.inf]; sigmaG_frac = 0.0333   # fitting where sigmaG is a % of the value
# fn_comments = '_alphan2p2_Gerr1percent'; alim = [-2, 2]; sigmaG_frac = 0.01   # fitting where sigmaG is a % of the value
# fn_comments = '_alphaninfpinf_Gerr1percent'; alim = [-np.inf, np.inf]; sigmaG_frac = 0.01   # fitting where sigmaG is a % of the value
# fn_comments = '_alphaninfpinf_Gerr2p4percent'; alim = [-np.inf, np.inf]; sigmaG_frac = 0.024   # 2.4% = average std/mean GTES for bolos 26 and 7
# fn_comments = '_alphan2p2_Gerr2p4percent'; alim = [-2, 2]; sigmaG_frac = 0.024   # 2.4% = average std/mean GTES for bolos 26 and 7
# fn_comments = '_alphan1p1_Gerr2p4percent'; alim = [-1, 1]; sigmaG_frac = 0.024   # 2.4% = average std/mean GTES for bolos 26 and 7
# fn_comments = '_alphaninfpinf_Gerr0p8percent'; alim = [-np.inf, np.inf]; sigmaG_frac = 0.008   # 0.8% = std/mean GTES of bolo 7
# fn_comments = '_alphan2p2_Gerr0p8percent'; alim = [-2, 2]; sigmaG_frac = 0.008   # 0.8% = std/mean GTES of bolo 7
# fn_comments = '_alphan1p1_Gerr0p8percent'; alim = [-1, 1]; sigmaG_frac = 0.008   # 0.8% = std/mean GTES of bolo 7
# fn_comments = '_alphaninfpinf_varythickness_0percent'; alim = [-np.inf, np.inf]   # 0.8% = std/mean GTES of bolo 7
# fn_comments = '_alphaninfpinf_varythickness_0p1percent'; alim = [-np.inf, np.inf]   # 0.8% = std/mean GTES of bolo 7
# fn_comments = '_alphaninfpinf_varythickness_1percent'; alim = [-np.inf, np.inf]   # 0.8% = std/mean GTES of bolo 7
# fn_comments = '_alphaninfpinf_varythickness_5percent'; alim = [-np.inf, np.inf]   # 0.8% = std/mean GTES of bolo 7
# fn_comments = '_alphaninfpinf_varythickness_0p001percent'; alim = [-np.inf, np.inf]   # 0.8% = std/mean GTES of bolo 7
fn_comments = '_alphaninfpinf_IstackI1d_350nm'; alim = [-np.inf, np.inf]   # 0.8% = std/mean GTES of bolo 7

# if constrained:
#     fn_comments = fn_comments_a01; alim = alim_a01
# else:
#     fn_comments = fn_comments_a0inf; alim = alim_a0inf
# fn_comments = fn_comments_a0inf; alim = alim_a0inf

plot_comments = ''
vmax = 1E4   # quality plot color bar scaling
# sigmaG_frac = 0.024   # sigma_G = x fraction of G
calc = 'median'   # how to evaluate fit parameters from simluation data
qplim = [-2,2]
bolo1b = True   # add bolo1 data to legacy prediction comparison plot

vary_thickness = False   # vary film thicknesses during simulation
derr = 0.00   # error on layer thicknesses [fraction of layer d]

# initial guess for fitter
p0_a0inf = np.array([0.7, 0.75, 1.2, 0.75, 0.1, 1.5])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]
p0_a01 = np.array([0.5, 0.5, 1, 1., 1., 1.])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]
p0 = p0_a0inf

# choose GTES data 
# ydata_fitGexplicit = [13.95595194, 5.235218152, 8.182147122, 10.11727864, 17.47817158, 5.653424631, 15.94469664, 3.655108238]   # pW/K at 170 mK fitting for G explicitly, weighted average on most bolos (*) (second vals have extra bling); bolo 1b, 24*, 23*, 22, 21*, 20, 7*, 13* 
# sigma_fitGexplicit = [0.073477411, 0.01773206, 0.021512022, 0.04186006, 0.067666665, 0.014601341, 0.083450365, 0.013604177]   # fitting for G explicitly produces very small error bars
ydata_fitGexplicit_nobling = np.array([13.95595194, 4.721712381, 7.89712938, 10.11727864, 17.22593561, 5.657104443, 15.94469664, 3.513915367])   # pW/K at 170 mK fitting for G explicitly, weighted average only on 7; bolo 1b, 24, 23, 22, 21, 20, 7*, 13 
sigma_fitGexplicit_nobling = [0.073477411, 0.034530085, 0.036798694, 0.04186006, 0.09953389, 0.015188074, 0.083450365, 0.01762426]
# sigma_percentG = sigmaG_frac*ydata_fitGexplicit_nobling
ydata = np.array(ydata_fitGexplicit_nobling); sigma = np.array(sigma_fitGexplicit_nobling)

bolos = np.array(['bolo 1b', 'bolo 24', 'bolo 23', 'bolo 22', 'bolo 21', 'bolo 20', 'bolo 7', 'bolo 13'])
bounds = [(0, np.inf), (0, np.inf), (0, np.inf), (alim[0], alim[1]), (alim[0], alim[1]), (alim[0], alim[1])]   # bounds for 6 fit parameters: G_U, G_W, G_I, alpha_U, alpha_W, alpha_I
plot_dir = analysis_dir + 'plots/layer_extraction_analysis/'
sim_file = analysis_dir + 'Analysis_Files/sim' + fn_comments + '.pkl'
csv_file = analysis_dir + 'Analysis_Files/sim' + fn_comments + '.csv'
data = [ydata, sigma] 

# bolos 1a-f vary leg length, all four legs have full microstrip
ydatavl_lb = np.array([22.17908389, 13.95595194, 10.21776418, 8.611287109, 7.207435165, np.nan]); sigmavl_lb = np.array([0.229136979, 0.073477411, 0.044379343, 0.027206631, 0.024171663, np.nan])   # pW/K, bolos with minimal bling
ydatavl_mb = np.array([22.19872947, np.nan, 10.64604145, 8.316849305, 7.560603448, 6.896700236]); sigmavl_mb = np.array([0.210591249, np.nan, 0.065518258, 0.060347632, 0.051737016, 0.039851469])   # pW/K, bolos with extra bling
ll_vl = np.array([120., 220., 320., 420., 520., 620.])*1E-3   # mm
llvl_all = np.append(ll_vl, ll_vl); ydatavl_all = np.append(ydatavl_lb, ydatavl_mb); sigmavl_all = np.append(sigmavl_lb, sigmavl_mb)
vlength_data = np.stack([ydatavl_all, sigmavl_all, llvl_all*1E3])

if latex_fonts:   # fonts for paper plots
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=15)
    plt.rc('font', weight='normal')
    plt.rcParams['text.latex.preamble']="\\usepackage{amsmath}"
    plt.rcParams['xtick.major.size'] = 5; plt.rcParams['xtick.minor.visible'] = False    
    plt.rcParams['ytick.major.size'] = 5; plt.rcParams['ytick.minor.visible'] = False

### Execute Analysis
if run_sim:   # run simulation with just hand-written function
    # sim_results = runsim_chisq(n_its, p0, data, bounds, plot_dir, show_yplots=show_plots, save_figs=save_figs, save_sim=save_sim, sim_file=sim_file, fn_comments=fn_comments)  
    sim_results = runsim_chisq(n_its, p0, data, bounds, plot_dir, show_yplots=show_plots, save_figs=save_figs, save_sim=save_sim, sim_file=sim_file, fn_comments=fn_comments, vary_thickness=vary_thickness, derr=derr)  


if quality_plots:   # plot G_x vs alpha_x parameter space with various fit results

    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
    sim_dataT = sim_dict['sim']; sim_data = sim_dataT.T   # simulation parameter values
    param_labels = ['G$_U$', 'G$_W$', 'G$_I$', '$\\alpha_U$', '$\\alpha_W$', '$\\alpha_I$']

    # if np.isinf(alim[1]):   # quality plot title
    #     qp_title = '$\\boldsymbol{\\mathbf{\\alpha \\in ['+str(alim[0])+',\infty)}}$'   # title for 1x3 quality plots
    # else:
    #     # qp_title = '$\\boldsymbol{\\mathbf{\\alpha \in [0,'+str(alim[1])+']}}$'   # title for 1x3 quality plots
    #     qp_title = '$\\boldsymbol{\\mathbf{\\alpha \in ['+str(alim[0])+','+str(alim[1])+']}}$'   # title for 1x3 quality plots
    qp_title = '$\\boldsymbol{\\mathbf{\\alpha \in ['+str(alim[0])+','+str(alim[1])+']}}$'   

    ### plot fit in 2D parameter space, take mean values of simulation
    results_mean = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_mean', title=qp_title+'\\textbf{ (Mean)}', vmax=vmax, calc='mean', qplim=qplim)
    params_mean, paramerrs_mean, kappas_mean, kappaerrs_mean, Gwire_mean, sigmaGwire_mean, chisq_mean = results_mean

    ### plot fit in 2D parameter space, take median values of simulation
    results_med = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_median', title=qp_title+'\\textbf{ (Median)}', vmax=vmax, calc='median', qplim=qplim)
    params_med, paramerrs_med, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_med

    ### pairwise correlation plots
    pairfig = pairwise(sim_data, param_labels, title=qp_title, save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments)

    # ## compare with unconstrained
    # sim_file_a01 = analysis_dir + 'Analysis_Files/sim' + fn_comments_a01 + '.pkl'
    # with open(sim_file_a01, 'rb') as infile:   # load simulation pkl
    #     sim_dict = pkl.load(infile)
    # alim = alim_a01
    # if np.isinf(alim[1]):   # quality plot title
    #     qp_title = '$\\boldsymbol{\\mathbf{\\alpha \\in [0,\infty)}}$'   # title for 1x3 quality plots
    # else:
    #     qp_title = '$\\boldsymbol{\\mathbf{\\alpha \in [0,'+str(alim[1])+']}}$'   # title for 1x3 quality plots
    # results_01 = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_mean', title=qp_title+'\\textbf{ (Mean)}', vmax=vmax, calc='mean')

    if save_csv:   # save model results to CSV file
        vals_mean = np.array([params_mean[0], params_mean[1], params_mean[2], params_mean[3], params_mean[4], params_mean[5], kappas_mean[0], kappas_mean[1], kappas_mean[2], Gwire_mean, chisq_mean])
        vals_med = np.array([params_med[0], params_med[1], params_med[2], params_med[3], params_med[4], params_med[5], kappas_med[0], kappas_med[1], kappas_med[2], Gwire_med, chisq_med])
        vals_err = np.array([paramerrs_mean[0], paramerrs_mean[1], paramerrs_mean[2], paramerrs_mean[3], paramerrs_mean[4], paramerrs_mean[5], kappaerrs_mean[0], kappaerrs_mean[1], kappaerrs_mean[2], sigmaGwire_mean, ''])   # should be the same for mean and median
        
        # write CSV     
        csv_params = np.array(['GU (pW/K)', 'GW (pW/K)', 'GI (pW/K)', 'alphaU', 'alphaW', 'alphaI', 'kappaU (pW/K/um)', 'kappaW (pW/K/um)', 'kappaI (pW/K/um)', 'Gwire (pW/K)', 'Chi-sq val'])
        fields = np.array(['Parameter', 'Mean', 'Median', 'Error'])  
        rows = [[csv_params[rr], vals_mean[rr], vals_med[rr], vals_err[rr]] for rr in np.arange(len(csv_params))]
        with open(csv_file, 'w') as csvfile:  
            csvwriter = csv.writer(csvfile)  # csv writer object  
            csvwriter.writerow(fields)  
            csvwriter.writerows(rows)


if lit_compare:

    ### load fit results
    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
    # if 'inf' in sim_file:
    #     sim_results = [np.median(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)]   # take median value instead of mean for alpha=[0,inf] model
    # else:
    #     sim_results = [sim_dict['fit']['fit_params'], sim_dict['fit']['fit_std']]
    # sim_dataT = sim_dict['sim']; sim_data = sim_dataT.T
    if calc=='mean':
        print('\nCalculating fit parameters as the mean of the simulation values.\n')
        sim_results = [np.mean(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)]
    elif calc=='median':
        print('\nCalculating fit parameters as the median of the simulation values.\n')
        sim_results = [np.median(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)]
    else:
        print('Unknown parameter calculation method. Choose "mean" or "median".')


    Gmeas_U, Gmeas_W, Gmeas_I, alpham_U, alpham_W, alpham_I = sim_results[0]; sigGU, sigGW, sigGI, sigalpha, sigalphaW, sigalphaI = sim_results[1]
    kappam_U = GtoKappa(Gmeas_U, A_U, L); sigkappam_U = GtoKappa(sigGU, A_U, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
    kappam_W = GtoKappa(Gmeas_W, A_W, L); sigkappam_W = GtoKappa(sigGW, A_W, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
    kappam_I = GtoKappa(Gmeas_I, A_I, L); sigkappam_I = GtoKappa(sigGI, A_I, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants

    ### Nb
    TD_Nb = 275   # K, Nb
    T_bath = 0.170   # measurement temperature (Tc of TESs), K
    Tc_Nb = 9.2   # Nb, K
    # vF_Nb = 1.37E6   # Nb Fermi velocity (electron velocity), m/s
    vs_Nb = 3480   # phonon velocity is the speed of sound in Nb, m/s
    vt_Nb = 2200   # transverse sound speed in Nb, m/s, from Dalrymple 1986
    vl_Nb = 5100   # longitudinal sound speed in Nb, m/s, from Dalrymple 1986

    kappapmfp_Nb = kappa_permfp(0.170, material='Nb')   # pW/K/um^2

    kappa_F17A = 10   # pw/K/um; Feshchenko et al 2017 sample A (L=5um), 200 nm thick, 1 μm wide and 5, 10 and 20 μm long Nb thin films
    kappa_F17B = 30   # pw/K/um; Feshchenko et al 2017 sample A (L=10um), 200 nm thick, 1 μm wide and 5, 10 and 20 μm long Nb thin films
    kappa_F17C = 100   # pw/K/um; Feshchenko et al 2017 sample A (L=20um), 200 nm thick, 1 μm wide and 5, 10 and 20 μm long Nb thin films


    print('\n'); print('Measured kappa_W: ', round(kappam_W, 2), ' +/- ', round(sigkappam_W, 2), ' pW/K/um')
    print('Predicted Nb kappa/mfp: ',  round(kappapmfp_Nb, 2), ' pW/K/um^2')
    print('Nb kappa/sqrt(2)*d: ',  round(kappam_W/(np.sqrt(2)*.400), 2), ' +/- ', round(sigkappam_W/(np.sqrt(2)*.400), 2), ' pW/K/um^2')
    print('W mfp from theory: ', round(kappam_W/kappapmfp_Nb, 2), ' +/- ', round(sigkappam_W/kappapmfp_Nb, 2), ' um'); print('\n')

    print("Measured W kappa/sqrt(2)*d = ", round(kappam_W/(np.sqrt(2)*.400), 2), " pW/K/um^2 assuming mfp = sqrt(2)*d")
    print("F17 Sample A Nb kappa/sqrt(2)*d: ", round(kappa_F17A/(np.sqrt(2)*.2), 2), ' pW/K/um^2')
    print("F17 Sample B Nb kappa/sqrt(2)*d: ", round(kappa_F17B/(np.sqrt(2)*.2), 2), ' pW/K/um^2')
    print("F17 Sample C Nb kappa/sqrt(2)*d: ", round(kappa_F17C/(np.sqrt(2)*.2), 2), ' pW/K/um^2'); print('\n')

    print('W mfp from theory: ', round(kappam_W/kappapmfp_Nb, 2), ' +/- ', round(sigkappam_W/kappapmfp_Nb, 2), ' um')
    print('F17 Sample A mfp from theory: ', round(kappa_F17A/kappapmfp_Nb, 5), ' um')
    print('F17 Sample B mfp from theory: ', round(kappa_F17B/kappapmfp_Nb, 2), ' um')
    print('F17 Sample C mfp from theory: ', round(kappa_F17C/kappapmfp_Nb, 2), ' um'); print('\n')


    ### SiN
    vs_SiN = 6986   # m/s; Wang et al
    TD_Si = 645   # K, silicon, low temp limit

    kappapmfp_Debye = kappa_permfp(0.170, material='SiN')   # Debye model
    kappapmfp_I = kappam_I / (np.sqrt(2)*.4)   # pW / K / um^2
    kappapmfp_U = kappam_I / (np.sqrt(2)*.42)   # pW / K / um^2

    kappapmfp_wang = 1/3*(0.083*T_bath+0.509*T_bath**3)*vs_SiN   # kappa per mfp from Casimir, pW / K / um^2; Si3N4; volumetric heat capacity * average sound speed
    kappa_wang = kappapmfp_wang*6.58   # pW / K / um for 10 um width SiN beam
    G_wang = kappa_wang*640/(1*10)   # pW / K for 640 um x 10 um x 1um SiN beam
    mfpI_Debye = kappam_I/kappapmfp_Debye   # um
    mfpU_Debye = kappam_U/kappapmfp_Debye   # um
    mfpI_Wang = kappam_I/kappapmfp_wang   # um
    mfpU_Wang = kappam_U/kappapmfp_wang   # um

    kappa_LP = 1.58*T_bath**1.54*1E3   # W / K / um, Geometry III for SiN film from Leivo & Pekola 1998 (they say mfp = d)
    kappamfp_LP = kappa_LP/0.200   # W / K / um^2, Geometry III from Leivo & Pekola 1998 (they say mfp = d)

    kappa_HolmesA = 1E-8/(2.7E-5)*1E6   # pW/K/um; Holmes98 sample A 
    kappa_HolmesF = 4E-9/(2.7E-5)*1E6   # pW/K/um; Holmes98 sample F = same as A but 10nm Au film on top of SiN
    c_bulkSiN = .58*.170**3*1E-6   # pJ/K/um^3; bulk SiN specific heat at 170 mK referenced in Holmes98 p97

    print("Measured I kappa: ", round(kappam_I, 2), "+/-", round(sigkappam_I, 2), " pW/K/um; mfp = ", round(mfpI_Wang, 2), " um from Wang's kappa/mpf")
    print("Measured U kappa: ", round(kappam_U, 2), "+/-", round(sigkappam_U, 2), " pW/K/um; mfp = ", round(mfpU_Wang, 2), " um from Wang's kappa/mpf")
    print("Wang's 10um wide Si3N4 beam kappa: ", round(kappa_wang, 2), ' pW/K/um')
    print("L&P98's 25um wide SiN kappa: ", round(kappa_LP, 2), ' pW/K/um'); print('\n')

    print("Measured I kappa/mfp = ", round(kappapmfp_I, 2), " pW/K/um^2 assuming mfp = sqrt(2)*d")
    print("Measured U kappa/mfp = ", round(kappapmfp_U, 2), " pW/K/um^2 assuming mfp = sqrt(2)*d")
    print("Wang's Si3N4 kappa/mfp: ", round(kappapmfp_wang, 2), ' pW/K/um^2')
    print("L&P98's SiN kappa/mfp: ", round(kappamfp_LP, 2), ' pW/K/um^2'); print('\n')

    print("Measured I kappa/sqrt(2)*d = ", round(kappapmfp_I, 2), "+/-", round(sigkappam_I/(np.sqrt(2)*.4), 2), " pW/K/um^2 assuming mfp = sqrt(2)*d")
    print("Measured U kappa/sqrt(2)*d = ", round(kappapmfp_U, 2), "+/-", round(sigkappam_U/(np.sqrt(2)*.42), 2), " pW/K/um^2 assuming mfp = sqrt(2)*d")
    print("Wang's Si3N4 kappa/sqrt(2)*d: ", round(kappa_wang/(np.sqrt(2)*1), 2), ' pW/K/um^2')
    print("L&P98's SiN kappa/sqrt(2)*d: ", round(kappa_LP/(np.sqrt(2)*.2), 2), ' pW/K/um^2')
    print("Holmes98's sample A SiN kappa/sqrt(2)*d: ", round(kappa_HolmesA/(np.sqrt(2)*1), 2), ' pW/K/um^2')
    print("Holmes98's sample F SiN kappa/sqrt(2)*d: ", round(kappa_HolmesF/(np.sqrt(2)*1), 2), ' pW/K/um^2'); print('\n')

    print('mfp_I from Debye: ', round(mfpI_Debye, 2), ' +/- ', round(sigkappam_I/kappapmfp_Debye, 2), ' um') 
    print('mfp_U from Debye: ', round(mfpU_Debye, 2), ' +/- ', round(sigkappam_U/kappapmfp_Debye, 2), ' um') 
    print("mfp_I from Wang", round(mfpI_Wang, 2), ' +/- ', round(sigkappam_I/kappapmfp_wang, 2), " um") 
    print("mfp_U from Wang", round(mfpU_Wang, 2), ' +/- ', round(sigkappam_U/kappapmfp_wang, 2), " um"); print('\n')

    ### dominant phonon wlength
    lambda_Nb = phonon_wlength(vs_Nb, T_bath)*1E6   # um
    lambda_SiN = phonon_wlength(vs_SiN, T_bath)*1E6   # um
    print("Dominant phonon wlength in W = ", round(lambda_Nb*1E3, 1), 'nm')
    print("Dominant phonon wlength in I&U = ", round(lambda_SiN*1E3, 1), 'nm'); print('\n')

    # calculate theoretical mfps
    mfp_HolmesA = 3*kappa_HolmesA/(c_bulkSiN*vs_SiN*1E6)   # sample A, bare SiN, leff = 3 kappa_eff / c vs
    mfp_HolmesF = 3*kappa_HolmesF/(c_bulkSiN*vs_SiN*1E6)   # sample F, 10 nm Au
    mfp_Cas = l_Casimir(7, .400)   # um, mfp in Casimir limit for d ~ w sample (not true for bolotest)

    vs_SiO = (2*5.8+3.7)/3*1E9   # um / s
    lSiO2_PLT = l_PLT02(vs_SiO)   # um

    mfpI_Wyb = l_bl(7, .400)   # um; boundary diffusive reflection limited phonon mfp (specular scattering will increase this)
    mfpU_Wyb = l_bl(7, .420)   # um; boundary diffusive reflection limited phonon mfp (specular scattering will increase this)

    # how does l_B vary with d?
    dtest = np.linspace(0.1,1)   # um
    wtest = 5;   # um
    lb_test = l_bl(wtest, dtest)
    p0=[2,1,.2]
    dparams, cv = curve_fit(monopower, dtest, lb_test, p0)
    plt.figure()
    plt.plot(dtest, lb_test, label='values')
    plt.plot(dtest, monopower(dtest, dparams[0], dparams[1], dparams[2]), label='power law fit', alpha=0.5)
    plt.xlabel('d [um]')
    plt.ylabel('l_B [um]')
    plt.legend()

    # how does l_B vary with w?
    dtest = 0.40   # um
    wtest = np.linspace(1,10);   # um
    p0=[1,0.5,0]
    lb_test = l_bl(wtest, dtest)
    wparams, cv = curve_fit(monopower, wtest, lb_test, p0)
    plt.figure()
    plt.plot(wtest, lb_test, label='values')
    plt.plot(wtest, monopower(wtest, wparams[0], wparams[1], wparams[2]), label='power law fit', alpha=0.5)
    plt.figure()
    plt.plot(wtest, lb_test)
    plt.xlabel('w [um]')
    plt.ylabel('l_B [um]')
    plt.xscale('log')
    plt.legend()


    mfpI_eff = l_eff(7, .4, f)   # um; reflection limited phonon mfp including spectral scattering (if f<1)
    gamma = mfpI_eff/mfpI_Wyb   # should be 1 if f=1 (Casimir limit)

    # reproduce Wyborne84 Fig 1
    dtest = np.ones(50); wtest = np.linspace(1,50)
    plt.figure(); plt.plot(wtest/dtest, (1-l_bl(wtest, dtest)/l_Casimir(wtest, dtest))*100)
    plt.xlabel('n'); plt.ylabel('1-l_b/l_C [%]')

    # reproduce Wyborne84 Fig 2
    ftest = np.linspace(0,1)
    w=6; d=.3
    plt.figure(); plt.plot(ftest, l_eff(w, d, ftest)/(l_bl(w, d)*np.ones(len(ftest))), '.')
    # plt.ylim(1,8.5)
    plt.ylabel('$\gamma$'); plt.xlabel('f')

    # comparison to radiative transport G
    GU_Holmes = G_Holmes(7*.420, .170)   # pW/K, fully ballistic G estimate for dielectrics
    GI_Holmes = G_Holmes(7*.400, .170)   # pW/K, fully ballistic G estimate for dielectrics
    xi_U = Gmeas_U/GU_Holmes   # reduction in G due to diffuse transport
    xi_I = Gmeas_I/GI_Holmes   # reduction in G due to diffuse transport

    GW_permfp = kappa_permfp(.170, material='Nb')*5*.400/220
    GI_permfp = kappa_permfp(.170, material='SiN')*7*.400/220
    GU_permfp = kappa_permfp(.170, material='SiN')*7*.420/220
    
    ### 2D vs 3D phonon dimensionality
    vt_SiN = 6.28E3   # transverse sound speed for Si3N4 at 0K, m/s, Bruls 2001
    vl_SiN = 10.3E3   # m/s, lateral sound speed for SiN
    dcrit_SiN = hbar * vt_SiN / (2*kB*T_bath) *1E6  # um
    dcrit_SiN_Holmes = 48 * vs_SiN*1E-4 / (T_bath*1E3)   # um if T is in mK, eqn 4.11 in Holmes thesis
    dcrit_Nb = hbar * vt_Nb / (2*kB*T_bath) *1E6  # um
    dcrit_Nb_Holmes = 48 * vs_Nb*1E-4 / (T_bath*1E3)   # um if T is in mK, eqn 4.11 in Holmes thesis

    print("d/d_crit for U: ", round(0.420/dcrit_SiN, 2))
    print("d/d_crit for W: ", round(0.400/dcrit_Nb, 2))
    print("d/d_crit for I: ", round(0.400*1/dcrit_SiN, 2)); print('\n')  
    print("d/d_crit for smallest U layer: ", round(0.340/dcrit_SiN, 2))
    print("d/d_crit for smallest W layer: ", round(0.160/dcrit_Nb, 2))
    print("d/d_crit for smallest I layer: ", round(0.270*dcrit_SiN, 2)); print('\n')

    TCdim_U = hbar*vt_SiN/(2*kB*.340*1E-6)   # K, 2D/3D crossover temp for smallest layers
    TCdim_I = hbar*vt_SiN/(2*kB*.270*1E-6)
    TCdim_I2 = hbar*vt_SiN/(2*kB*.400*1E-6)
    TCdim_W = hbar*vt_Nb/(2*kB*.160*1E-6)
    TCdim_W2 = hbar*vt_Nb/(2*kB*.340*1E-6)

    print("T/T_crit for smallest U: ", round(0.170/TCdim_U, 2))
    print("T/T_crit for smallest W: ", round(0.170/TCdim_W, 2))
    print("T/T_crit for smallest I: ", round(0.170/TCdim_I, 2)); print('\n')  

    lwidth = 7*1E-6   # um
    def mstar(vs, vt, d):
        T2 = ((vs**2 - vt**2)/(3*vs**2))**(-1/2)
        return hbar/(2*d*vt)*T2

    def kappa2D(w, d, vt, vl, T=0.170):
        # lM = 1E-6   # m, super approximate mfp of "uncut membrane", interatomic distance in SiN
        C = 4*np.log(300)
        # C = 4*np.log(2*lM/(w))

        mstar = hbar/(2*d*vt) * ((vl**2-vt**2)/(3*vl**2))**(-1/2)   # effective mass, J*s/
        T1 = 1.202*(1/vt+1/vl)*(kB*T/(planck))**2
        T2 = np.sqrt(2*mstar/hbar)*5/16*1.341*(kB*T/(planck))**3/2
        # return 3*C*w**2*kB*np.pi*( T1 + T2 )
        return 3*C**kB*np.pi*( T1 + T2 )
    

    # estimate surface roughness from measured mfp
    def fspec(eta, T, vs):
        lambda_dom = phonon_wlength(vs, T)   # dominant phonon wavelength
        q = 2*np.pi/lambda_dom
        return np.exp(-4*np.pi * eta**2 * q**2)
    
    def eta_finder(eta, T, vs, f_meas):   # root finder for finding what eta fits with measured specular scattering probability f_meas
        return fspec(eta, T, vs) - f_meas
    
    mfp_W = kappam_W/kappapmfp_Nb   # um
    d_W = 0.400   # um
    fmeas_Nb = 1 - np.sqrt(2)*d_W/mfp_W
    vs_Nb = 3480   # phonon velocity is the speed of sound in Nb, m/s

    eta_test = 1   # guess for eta value, nm
    eta_W = fsolve(eta_finder, eta_test, args=(0.170, vs_Nb*1E9, fmeas_Nb))[0]   # in nm

    vs_SiN = 6986   # m/s; Wang et al
    mfp_I = mfpI_Wang   # um
    mfp_U = mfpU_Wang   # um
    d_I = 0.400   # um
    d_U = 0.420   # um
    fmeas_I = 1 - np.sqrt(2)*d_I/mfp_I
    fmeas_U = 1 - np.sqrt(2)*d_U/mfp_U
    eta_I = fsolve(eta_finder, eta_test, args=(0.170, vs_SiN*1E9, fmeas_I))[0]   # in nm
    eta_U = fsolve(eta_finder, eta_test, args=(0.170, vs_SiN*1E9, fmeas_U))[0]   # in nm

    print("W layer surface roughness: ", round(eta_W, 2), 'nm')  
    print("I layer surface roughness: ", round(eta_I, 2), 'nm') 
    print("U layer surface roughness: ", round(eta_U, 2), 'nm'); print('\n')  


if compare_legacy:   # compare G predictions with NIST legacy data

    # load fit 
    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
    simresults_mean = np.array([np.mean(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)])
    simresults_med = np.array([np.median(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)])

    # if np.isinf(alim[1]):
    #     plot_comments = '_unconstrained'
    #     title='$\\alpha \\in [0,\\infty)$ Model Predictions'
    # else:
    #     plot_comments = '_constrained'
    #     title='$\\alpha \\in [0,1]$ Model Predictions'
    title = '$\\boldsymbol{\\mathbf{\\alpha \in ['+str(alim[0])+','+str(alim[1])+']}}$ Model Predictions'   

    sim_dataT = sim_dict['sim']; sim_data = sim_dataT.T

    L = 220   # bolotest leg length, um
    boloGs = ydata; sigma_boloGs = sigma   # choose bolotest G's to compare
    wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20
    A_bolo = np.array([(7*4*.420+5*4*.160+3*4*.340+7*4*.350+7*4*.400), (7*1*.420+7*3*.340+5*.160+3*.340+7*.350+7*.400), (7*2*.420+7*2*.340+5*2*.160+3*2*.340+7*2*.350+7*2*.400), (7*3*.420+7*1*.340+5*3*.160+3*3*.340+7*3*.350+7*3*.400), (7*1*.420+7*3*.400+5*1*.160+3*1*.285+7*3*.370+7*1*.350), (7*4*.420+5*1*.160+wstack_width*3*.385+3*1*.285+7*1*.340), (7*3*.420+7*1*.400+5*3*.160+3*1*3.340+7*3*.350+7*1*.670+7*3*.400), (7*1*.420+7*3*.400+5*1*.160+3*1*.285+7*1*.350) ])   # bolotest areas
    AoL_bolo = A_bolo/L   # A/L for bolotest devices
    data1b=np.array([ydata[0], sigma[0]]) if bolo1b else []  # plot bolo1b data?

    predict_Glegacy(simresults_mean, data1b=data1b, save_figs=save_figs, title=title+' (Mean)', plot_comments=fn_comments+'_mean', fs=(7,7))
    predict_Glegacy(simresults_mean, data1b=data1b, save_figs=save_figs, title=title+' (Mean), $\\beta$=0.8', plot_comments=fn_comments+'_mean_beta0p8', fs=(7,7), Lscale=0.8)
    # predict_Glegacy(simresults_mean, save_figs=save_figs, plot_comments=plot_comments+'_forpaper', fs=(6.5,5))
    predict_Glegacy(simresults_med, data1b=data1b, save_figs=save_figs, title=title+' (Median)', plot_comments=fn_comments+'_median')
    predict_Glegacy(simresults_med, data1b=data1b, save_figs=save_figs, title=title+' (Median), $\\beta$=0.8', plot_comments=fn_comments+'_median_beta0p8', Lscale=0.8)
    # (G_layer(simresults_mean, dI1, layer='I') + G_layer(simresults_mean, dI2, layer='I')) *lw/7 *220/ll   # not sure what this was for?

    # title="Predictions from Layer $\kappa$'s"
    # plot_comments = '_kappa'
    # predict_Glegacy(simresults_mean, data1b=data1b, save_figs=save_figs, estimator='kappa', title=title+' (Mean)', plot_comments=plot_comments)

    
if design_implications:   # making plots to illustrate TES and NIS design implications of this work

    ### plot G_TES and TFN as a function of substrate width
    ### assumes two legs with full microstrip and two are 
    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
    simresults_mean = np.array([np.mean(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)])
    simresults_med = np.array([np.median(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)])
    sim_results = simresults_mean if calc=='mean' else simresults_med

    lwidths = np.linspace(2.75, 100/7, num=100)   # um
    # Glims = [1.9,34]
    # NEPlims = [2.1,7]
    Glims = [1.6,30]
    NEPlims = [2.1,9]

    # plot_GandTFNEP(sim_results, lwidths, save_fig=save_figs, plot_dir=plot_dir, plot_NEPerr=True, plot_Gerr=True, Glims=Glims, NEPlims=NEPlims, plotG=False)
    plot_GandTFNEP(sim_results, lwidths, save_fig=save_figs, plot_dir=plot_dir, plot_NEPerr=True, plot_Gerr=True, Glims=Glims, plotG=False)
    
    ### find crossover dsub value where G_micro = G_sub 
    # this will be constant with legnth but not constant with width since w_W1/W2 are constant
    # at lw=10 um, the crossover thickness is 1.17  um
    # at lw=7 um, the crossover thickness is 1.14  um
    # at lw=5 um, the crossover thickness is 1.21  um
    # at lw=41.5 um, the crossover thickness is 1.02  um

    lw_test = 7; ll_test = 220   # um; um
    fab='legacy'
    # dsubs = np.linspace(0, 2, int(1E4))  # um

    def xover_finder(dsub, fit, lw, ll, fab=fab):   # when G_U/G_micro = 1, returns 0
        G_micros = Gfrommodel(fit, dsub, lw, ll, layer='wiring', fab=fab)[0]
        G_Us = Gfrommodel(fit, dsub, lw, ll, layer='U', fab=fab)[0]
        return G_Us/G_micros - 1

    # ll_test = 1   # um, shouldn't matter since ratio is length independent
    # lwidths = np.arange(0, 50, 10)

    xover_dsub = fsolve(xover_finder, ll_test, args=(sim_results, lw_test, ll_test))[0]
    print('Substrate thickness above which G_substrate >= G_microstrip = '+str(round(xover_dsub, 1))+' um for leg width of '+str(lw_test)+' um and leg length of '+str(ll_test)+' um.')


if load_and_plot:   # for loading a simulation and replotting things; kinda scrap

    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
    # sim_results = [sim_dict['fit']['fit_params'], sim_dict['fit']['fit_std']]
    sim_data = sim_dict['sim']
    sim_results = [np.median(sim_data, axis=0), np.std(sim_data, axis=0)]   # take median value instead of mean
    fn_comments2 = fn_comments+plot_comments
    Gmeas_U, Gmeas_W, Gmeas_I, alpham_U, alpham_W, alpham_I = sim_results[0]; sigGU, sigGW, sigGI, sigalphaU, sigalphaW, sigalphaI = sim_results[1]

    Gwires = G_wirestack(sim_data.T)
    Gwire = np.median(Gwires); Gwire_std = np.std(Gwires)   # take median value instead of mean

    print('Results from Monte Carlo Sim - chisq Min')
    print('G_U(420 nm) = ', round(Gmeas_U, 2), ' +/- ', round(sigGU, 2), 'pW/K')
    print('G_W(400 nm) = ', round(Gmeas_W, 2), ' +/- ', round(sigGW, 2), 'pW/K')
    print('G_I(400 nm) = ', round(Gmeas_I, 2), ' +/- ', round(sigGI, 2), 'pW/K')
    print('alpha_U = ', round(alpham_U, 2), ' +/- ', round(sigalphaU, 2))
    print('alpha_W = ', round(alpham_W, 2), ' +/- ', round(sigalphaW, 2))
    print('alpha_I = ', round(alpham_I, 2), ' +/- ', round(sigalphaI, 2))
    print('')
    print('G_wirestack = ', round(Gwire, 2), ' +/- ', round(Gwire_std, 2), 'pW/K')
    print('')

    qualityplots(data, sim_results, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments2, title=qp_title)

    kappam_U = GtoKappa(Gmeas_U, A_U, L); sigkappam_U = GtoKappa(sigGU, A_U, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
    kappam_W = GtoKappa(Gmeas_W, A_W, L); sigkappam_W = GtoKappa(sigGW, A_W, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
    kappam_I = GtoKappa(Gmeas_I, A_I, L); sigkappam_I = GtoKappa(sigGI, A_I, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
    print('Kappa_U: ', round(kappam_U, 2), ' +/- ', round(sigkappam_U, 2), ' pW/K/um')
    print('Kappa_W: ', round(kappam_W, 2), ' +/- ', round(sigkappam_W, 2), ' pW/K/um')
    print('Kappa_I: ', round(kappam_I, 2), ' +/- ', round(sigkappam_I, 2), ' pW/K/um')

    chisq_fit = chisq_val(sim_results[0], data)
    print('chisq value for the fit: ', round(chisq_fit, 3)) 


if bimodal_solns:

    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
    sim_dataT = sim_dict['sim']; sim_data = sim_dataT.T   # simulation parameter values
    param_labels = ['G$_U$', 'G$_W$', 'G$_I$', '$\\alpha_U$', '$\\alpha_W$', '$\\alpha_I$']

    if np.isinf(alim[1]):   # quality plot title
        qp_title = '$\\boldsymbol{\\mathbf{\\alpha \\in [0,\infty)}}$'   # title for 1x3 quality plots
    else:
        qp_title = '$\\boldsymbol{\\mathbf{\\alpha \in [0,'+str(alim[1])+']}}$'   # title for 1x3 quality plots

    data1b=np.array([ydata[0], sigma[0]]) if bolo1b else []  # plot bolo1b data?

    ### pairwise correlation plots
    pairfig = pairwise(sim_data, param_labels, title=qp_title, save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments)

    # # ### analyze sub-populations of solutions 
    # aWlim = 1E-5; aUlim = 0.7   # limits to delineate two solution spaces
    # lowa = np.where((sim_data[4] < aWlim) & (sim_data[3] < aUlim))[0]
    # # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W<}}$ '+str(aWlim)+' and $\\boldsymbol{\\mathbf{\\alpha_U<}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_overplotlowa', indsop=lowa, oplotlabel='low $\\alpha$')
    # # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W<}}$ '+str(aWlim)+' and $\\boldsymbol{\\mathbf{\\alpha_U<}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_lowa', indstp=lowa)

    # higha = np.where((sim_data[4] > aWlim) | (sim_data[3] > aUlim))[0]   # hopefully this removes bimodal solutions
    # # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W>}}$ '+str(aWlim)+' or $\\boldsymbol{\\mathbf{\\alpha_U>}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_overplothigha', indsop=higha, oplotlabel='high $\\alpha$')
    # # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W>}}$ '+str(aWlim)+' or $\\boldsymbol{\\mathbf{\\alpha_U>}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_higha', indstp=higha)

    # print('\n\nAnalyzing only HIGH aW and aU solutions:')
    # results_higha = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_higha', title=qp_title+'\\textbf{, high aW and aU (Mean)}', vmax=vmax, calc='mean', spinds=higha)
    # params_higha, paramerrs_higha, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_higha
    # fit_higha = np.array([params_higha, paramerrs_higha])

    # print('\n\nAnalyzing only LOW aW and aU solutions:')
    # results_lowa = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_lowa', title=qp_title+'\\textbf{ low aW and aU (Mean)}', vmax=vmax, calc='mean', spinds=lowa)
    # params_lowa, paramerrs_lowa, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_lowa
    # fit_lowa = np.array([params_lowa, paramerrs_lowa])

    # ### compare legacy predictions

    # predict_Glegacy(fit_higha, data1b=data1b, save_figs=save_figs, title=qp_title+' (High Alpha)', plot_comments=plot_comments+'_higha', fs=(7,7))
    # predict_Glegacy(fit_lowa, data1b=data1b, save_figs=save_figs, title=qp_title+' (Low Alpha)', plot_comments=plot_comments+'_lowa')

    aWlim = 1.5E-4; GUlim = 0.63   # limits to delineate two solution spaces
    lowGU = np.where((sim_data[4] <= aWlim) & (sim_data[0] <= GUlim))[0]
    pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{G_U<}}$ '+str(GUlim)+' pW/K and $\\boldsymbol{\\mathbf{\\alpha_W<}}$ '+str(aWlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_overplotlowGU', indsop=lowGU, oplotlabel='low GU')
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W<}}$ '+str(aWlim)+' and $\\boldsymbol{\\mathbf{\\alpha_U<}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_lowGU', indstp=lowGU)

    highGU = np.where((sim_data[4] > aWlim) | (sim_data[0] > GUlim))[0]
    pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{G_U>}}$ '+str(GUlim)+' pW/K or $\\boldsymbol{\\mathbf{\\alpha_W>}}$ '+str(aWlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_overplotlowGU', indsop=highGU, oplotlabel='high GU')
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W>}}$ '+str(aWlim)+' or $\\boldsymbol{\\mathbf{\\alpha_U>}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_higha', indstp=higha)


    print('\n\nAnalyzing only LOW GU solutions:')
    results_lowGU = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_lowGU', title=qp_title+'\\textbf{, Low GU (Mean)}', vmax=vmax, calc='mean', spinds=lowGU)
    params_lowGU, paramerrs_lowGU, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_lowGU
    fit_lowGU = np.array([params_lowGU, paramerrs_lowGU])

    print('\n\nAnalyzing only HIGH GU solutions:')
    results_highGU = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_highGU', title=qp_title+'\\textbf{, High GU (Mean)}', vmax=vmax, calc='mean', spinds=highGU)
    params_highGU, paramerrs_highGU, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_highGU
    fit_highGU = np.array([params_highGU, paramerrs_highGU])

    # compare legacy predictions
    predict_Glegacy(fit_lowGU, data1b=data1b, save_figs=save_figs, title=qp_title+'\\textbf{ (Low GU)}', plot_comments=plot_comments+'_lowGU')
    predict_Glegacy(fit_highGU, data1b=data1b, save_figs=save_figs, title=qp_title+'\\textbf{ (High GU)}', plot_comments=plot_comments+'_highGU', fs=(7,7))

    plot_modelvdata(fit_lowGU, data, title=qp_title+'\\textbf{ (Low GU)}', vlength_data=vlength_data, plot_bolotest=False)
    plot_modelvdata(fit_highGU, data, title=qp_title+'\\textbf{ (High GU)}', vlength_data=vlength_data, plot_bolotest=False)


if compare_modelanddata:

    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
    sim_dataT = sim_dict['sim']; sim_data = sim_dataT.T   # simulation parameter values
    simresults_mean = np.array([np.mean(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)])
    simresults_med = np.array([np.median(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)])

    # if np.isinf(alim[1]):   # quality plot title
    #     title = '$\\boldsymbol{\\mathbf{\\alpha \\in [0,\infty)}}$'   # plot title
    # else:
    #     title = '$\\boldsymbol{\\mathbf{\\alpha \in [0,'+str(alim[1])+']}}$'   # plot title
    title = '$\\boldsymbol{\\mathbf{\\alpha \in ['+str(alim[0])+','+str(alim[1])+']}}$ Model Predictions'   

    plot_modelvdata(simresults_mean, data, title=title+' (Mean)')

    # # check subsections of solutions
    # aWlim = 1.5E-4; GUlim = 0.63   # limits to delineate two solution spaces
    # lowGU = np.where((sim_data[4] < aWlim) & (sim_data[0] < GUlim))[0]
    # highGU = np.where((sim_data[4] > aWlim) | (sim_data[0] > GUlim))[0]

    # results_highGU = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_highgu', title=title+'\\textbf{, high aW or aU (Mean)}', vmax=vmax, calc='mean', spinds=highGU, plot=False)
    # params_highGU, paramerrs_highGU, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_highGU
    # fit_highGU = np.array([params_highGU, paramerrs_highGU])

    # # print('\n\nAnalyzing only LOW aW and aU solutions:')
    # results_lowGU = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_lowGU', title=title+'\\textbf{ low aW and aU (Mean)}', vmax=vmax, calc='mean', spinds=lowGU, plot=False)
    # params_lowGU, paramerrs_lowGU, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_lowGU
    # fit_lowGU = np.array([params_lowGU, paramerrs_lowGU])

    # plot_modelvdata(fit_lowGU, data, title=title+', Low GU Solns')
    # plot_modelvdata(fit_highGU, data, title=title+', High GU Solns')

if analyze_vlengthdata:

    # how does G vary with L?
    # p0=[1,-0.5,25]
    # params_lb, t_lb, sigmat_lb = fit_power(ll_vl, ydatavl_lb, p0, sigma=sigmavl_lb, absolute_sigma=True)
    # # params_mb, t_mb, sigmat_mb = fit_power(ll_vl, ydatavl_mb, p0, sigma=sigmavl_mb, absolute_sigma=True)
    # params_all, t_all, sigmat_all = fit_power(llvl_all, ydatavl_all, p0, sigma=sigmavl_all, absolute_sigma=True)

    # ll_toplot = np.logspace(-1., 0)
    # plt.figure()
    # plt.errorbar(ll_vl*1E3, ydatavl_lb, yerr=sigmavl_lb, label='Less BLING', linestyle='None', capsize=3, capthick=2, alpha=0.7)
    # plt.errorbar(ll_vl*1E3, ydatavl_mb, yerr=sigmavl_mb, label='More BLING', linestyle='None', capsize=3, capthick=2, alpha=0.7, color='C3')
    # plt.plot(ll_toplot*1E3, monopower(ll_toplot, params_lb[0], params_lb[1], params_lb[2]), alpha=0.5, color='C0')
    # # plt.plot(ll_toplot*1E3, monopower(ll_toplot, params_mb[0], params_mb[1], params_mb[2]), alpha=0.5, color='C3')
    # plt.ylabel('G [pW/K]'); plt.xlabel('Leg Length [um]')
    # plt.legend()
    # plt.yscale('log'); plt.xscale('log')
    # plt.annotate('$\\beta={t} \\pm {err}$'.format(t=round(-t_lb, 2), err=round(sigmat_lb, 2)), (220, 15), color='C0', fontsize=15, weight='bold')
    # # plt.annotate('$\\beta={t} \\pm {err}$'.format(t=round(-t_mb, 2), err=round(sigmat_mb, 2)), (320, 11.25), color='C3', fontsize=15, weight='bold')
    # plt.title('G$~$L$^{-\\beta}$ -- $\\beta_{all}='+str(round(-t_all, 2))+' \\pm '+str(round(sigmat_all, 2))+'$')

    # compare to model predictions
    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
    sim_dataT = sim_dict['sim']; sim_data = sim_dataT.T   # simulation parameter values
    simresults_mean = np.array([np.mean(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)])
    # simresults_med = np.array([np.median(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)])

    title = 'Predictions for Bolotest Data, '   
    plot_modelvdata(simresults_mean, data, title=title+' $\\sigma_G$ = 0.8\\% G', plot_bolotest=True)
    # plot_modelvdata(simresults_mean, data, title=title+' Bolos 1a-f', vlength_data=vlength_data, plot_bolotest=False)
    # plot_modelvdata(simresults_mean, data, title=title+' Bolos 1a-f, $\\beta=0.8$', vlength_data=vlength_data, plot_bolotest=False, Lscale=0.8)


if byeye:

    params_byeye = np.array([0.7, 0.8, 1.3, 1, 0.5, 1.2])
    sigma_params = np.array([0, 0, 0, 0, 0, 0])
    fit = np.array([params_byeye, sigma_params])
    title = 'Hand-chosen fit parameters '   
    data1b = np.array([ydata[0], sigma[0]]) if bolo1b else []  # plot bolo1b data?

    results = qualityplots(data, params_byeye, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments, title=title, vmax=1E3, calc='mean', qplim=qplim)

    predict_Glegacy(fit, data1b=data1b, save_figs=save_figs, title=title, plot_comments=fn_comments, fs=(7,7))
    plot_modelvdata(fit, data, title=title+' L=220 um Bolos', plot_bolotest=True)
    plot_modelvdata(fit, data, title=title+' Bolos 1a-f and bolotest', vlength_data=vlength_data, plot_bolotest=False)
    plot_modelvdata(fit, data, title=title+' Bolos 1a-f and bolotest, $\\beta=0.8$', vlength_data=vlength_data, plot_bolotest=False, Lscale=0.8)


    # testdd0 = np.linspace(0.01, 2)
    # plt.figure()
    # plt.plot(testdd0, testdd0**1, label='alpha=0')
    # plt.plot(testdd0, testdd0**1.5, label='alpha=0.5')
    # plt.plot(testdd0, testdd0**2, label='alpha=1')
    # plt.plot(testdd0, testdd0**2.5, label='alpha=1.5')
    # plt.legend()
    # plt.xlabel('d/d0')

if scrap:

    # f = 0.9   # fraction of diffuse reflections; 1 = Casimir limit
    # n = 1
    # Jtest = np.arange(10)
    # Jtest = np.array([1E-10, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # kdtest = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # funcvals = [sumfunc_J(n, f, Jt) for Jt in Jtest]
    # firstvals = [firstterm(n, Jt) for Jt in Jtest]
    # secondvals = [secondterm(n, Jt, kdtest[jj]) for jj, Jt in enumerate(Jtest)] 

    # reproduce Wyborne84 Fig 1 - this is correct
    # dtest = np.ones(50); wtest = np.linspace(1,50)
    # plt.figure(); plt.plot(wtest/dtest, (1-l_bl(wtest, dtest)/l_Casimir(wtest, dtest))*100)
    # plt.xlabel('n'); plt.ylabel('1-l_b/l_C [%]')
    # plt.xlim(0,50); plt.ylim(0,60)


    # wtest = 7   # um
    # dtest = 0.4   # um
    # ftest = 0.9
    # # leff = l_eff(wtest, dtest, ftest, sumlim=100)   # um
    # lb_test = l_bl(wtest, dtest)

    # f_W = np.linspace(0,1)
    # ftest = np.array([0.05, 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])
    # w_W=6; d_W=.3   # um, beam dimensions in Wybourne 84
    # lb = l_bl(w_W, d_W)*np.ones(len(f_W))   # um
    # leff_W = [l_eff(w_W, d_W, ff, sumlim=100)  for ff in f_W] # um
    # leff_test = [l_eff(w_W, d_W, ff, sumlim=100)  for ff in ftest] # um
    # leff_test = [l_eff(wtest, dtest, ff, sumlim=100)  for ff in ftest] # um
    # leff_W = l_eff(w_W, d_W, f_W, sumlim=100)   # um
    # plt.figure(); plt.plot(f_W, leff_W, '.')

    # plt.figure(); plt.plot(ftest, l_eff(w_W, d_W, ftest)/lb, '.')
    # # plt.ylim(1,8.5)
    # plt.ylabel('$\gamma$'); plt.xlabel('f')

    ftest = np.array([0.01, 0.2, 0.4, 0.6,  0.8, 1])
    # wtest=6; dtest=6   # um, n=1
    ntest = np.array([1, 5, 10])   # aspect ratio
    # ntest = 1   # aspect ratio
    # ntest = np.array([1])   # aspect ratio
    dtest = 1; wtest = dtest*ntest   # um
    # ntest = wtest/dtest
    # Jtest = 0
    # sumfunc_J(1, 0.1, Jtest)   # this increases with increasing f, regardless of J
    # firstterm(1, Jtest)   # this increases with increasing f, regardless of J
    leff_test = np.array([l_eff(wtest, dtest, ff, sumlim=1E4) for ff in ftest]) # um
    lb = l_bl(wtest, dtest)   # um

    
    plt.figure()
    lineObjects = plt.plot(ftest, leff_test/lb, '.')
    # plt.plot(ftest, leff_test/min(leff_test), '.', label='Normalized leff')
    # plt.ylim(1,8.5)
    plt.ylabel('$\gamma$'); plt.xlabel('f')
    plt.legend(iter(lineObjects), ['n = {}'.format(nt) for nt in ntest])

    # def prefactor(f, J):
    #     return f * (1-f)**(J)
    #     # return (f*(1-f))**J
    
    # plt.figure()
    # plt.plot(ftest, prefactor(ftest, 0), label='J=0')
    # plt.plot(ftest, prefactor(ftest, 1), label='J=1')
    # plt.plot(ftest, prefactor(ftest, 2), label='J=2')
    # plt.plot(ftest, prefactor(ftest, 3), label='J=3')
    # plt.plot(ftest, prefactor(ftest, 4), label='J=4')
    # plt.plot(ftest, prefactor(ftest, 10), label='J=10')
    # plt.xlabel('f'); plt.ylabel('Prefactor Value Evaluated at J')
    # plt.legend()

    # plt.figure()
    # plt.plot(ftest, np.sum([prefactor(ftest, JJ) for JJ in np.arange(0, 1)], axis=0), label='Sum up to J=0')
    # plt.plot(ftest, np.sum([prefactor(ftest, JJ) for JJ in np.arange(0,2)], axis=0), label='Sum up to J=1')
    # plt.plot(ftest, np.sum([prefactor(ftest, JJ) for JJ in np.arange(0,3)], axis=0), label='Sum up to J=2')
    # plt.plot(ftest, np.sum([prefactor(ftest, JJ) for JJ in np.arange(0,5)], axis=0), label='Sum up to J=5')
    # plt.plot(ftest, np.sum([prefactor(ftest, JJ) for JJ in np.arange(0,10)], axis=0), label='Sum up to J=10')
    # plt.plot(ftest, np.sum([prefactor(ftest, JJ) for JJ in np.arange(0,100)], axis=0), label='Sum up to J=100')
    # plt.plot(ftest, np.sum([prefactor(ftest, JJ) for JJ in np.arange(0,1000)], axis=0), label='Sum up to J=1000')
    # plt.plot(ftest, np.sum([prefactor(ftest, JJ) for JJ in np.arange(0,1E4)], axis=0), label='Sum up to J=1E4')
    # plt.legend()
    # plt.xlabel('f'); plt.ylabel('Prefactor Value Cumulative Sum')
    # plt.title('Sum of f(1-f)$^J$ over J')

    # sumlims = np.array([1, 1E1, 1E2, 1E3, 1E4, 1E5])

    # sumvals1 = np.array([np.sum([(firstterm(1, JJ) + secondterm(1,JJ)) for JJ in np.arange(0,sl)], axis=0) for sl in sumlims])
    # sumvals5 = np.array([np.sum([(firstterm(5, JJ) + secondterm(5, JJ)) for JJ in np.arange(0,sl)], axis=0) for sl in sumlims])
    # sumvals10 = np.array([np.sum([(firstterm(10, JJ) + secondterm(10, JJ)) for JJ in np.arange(0,sl)], axis=0) for sl in sumlims])
    # sumvals20 = np.array([np.sum([(firstterm(20, JJ) + secondterm(20, JJ)) for JJ in np.arange(0,sl)], axis=0) for sl in sumlims])
    # plt.figure()
    # plt.plot(sumlims, sumvals1, label='n=1')
    # plt.plot(sumlims, sumvals5, label='n=5')
    # plt.plot(sumlims, sumvals10, label='n=10')
    # plt.plot(sumlims, sumvals20, label='n=20')
    # plt.xlabel('Max J in Summation'); plt.ylabel('Summed J Values')
    # plt.legend()
    # plt.yscale('log')

    # leffvals1 = np.array([l_eff(wtest, dtest, 0.99, sumlim=sl) for sl in sumlims])
    # leffvals05 = np.array([l_eff(wtest, dtest, 0.5, sumlim=sl) for sl in sumlims])
    # leffvals0 = np.array([l_eff(wtest, dtest, 0.01, sumlim=sl) for sl in sumlims])
    # plt.figure()
    # plt.plot(sumlims, leffvals1, label='f=0.99')
    # plt.plot(sumlims, leffvals05, label='f=0.5')
    # plt.plot(sumlims, leffvals0, label='f=0.01')
    # plt.xlabel('Max J in Summation'); plt.ylabel('leff Values')
    # plt.legend()
    # plt.yscale('log')

    # # Jrange = np.array([0, 1, 10, 50, 100, 500, 1000, 5000, 1E4, 5E4, 1E5])
    # Jrange = np.array([0, 1, 10, 50, 100])
    # termvals1 = np.array([(firstterm(1, JJ) + secondterm(1, JJ)) for JJ in Jrange])
    # # termvals2 = np.array([(firstterm(2, JJ) + secondterm(2, JJ)) for JJ in Jrange])
    # termvals10 = np.array([(firstterm(10, JJ) + secondterm(10, JJ)) for JJ in Jrange])
    # termvals20 = np.array([(firstterm(20, JJ) + secondterm(20, JJ)) for JJ in Jrange])
    # termvals50 = np.array([(firstterm(50, JJ) + secondterm(50, JJ)) for JJ in Jrange])
    # plt.figure()
    # plt.plot(Jrange, termvals1, '.', label='n=1')
    # # plt.plot(Jrange, termvals2, '.', label='n=2')
    # plt.plot(Jrange, termvals10, '.', label='n=10')
    # plt.plot(Jrange, termvals20, '.', label='n=20')
    # plt.plot(Jrange, termvals50, '.', label='n=50')
    # plt.xlabel('J'); plt.ylabel('First Term + Second Term at J')
    # # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()

plt.show()