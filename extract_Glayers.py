"""
Script for measuring individual layer contributions to the total thermal conductivity of a TES. 
Data from 'bolotest' measurements run in 2018. 
G_total and error is from TES power law fit measured at Tc and scaled to 170 mK, assuming dP/dT = G = n*k*Tc^(n-1).
Layers: U = SiN + SiO substrate, W = Nb wiring layers (<=2 layers per leg), I = SiN insulating layers (<=2 layers per leg)

aharkehosemann@gmail.com
2022/01


UPDATES
2023/07/12: quote mean values, note bimodality and put total population in SI but note two populations share the same chi-squared

TODO: redo six-layer fit to compare chi-squared values?
"""
from bolotest_routines import *
from scipy.optimize import fsolve
import csv 


### User Switches
# choose analysis
run_sim = False   # run MC simulation for fitting model
quality_plots = False   # results on G_x vs alpha_x parameter space for each layer
random_initguess = False   # try simulation with randomized initial guesses
average_qp = False   # show average value for 2D quality plot over MC sim to check for scattering
lit_compare = False   # compare measured conductivities with literature values
compare_legacy = False   # compare with NIST sub-mm bolo legacy data
design_implications = False
load_and_plot = False   # scrap; currently replotting separate quality plots into a 1x3 subplot figure
scrap = False
bimodal_solns = True
compare_modelanddata = False   # plot model predictions and bolotest data

# options
save_figs = False   
save_sim = False   # save full simulation
save_csv = False   # save csv file of resulting parameters
show_plots = False   # show simulated y-data plots during MC simulation
calc_Gwire = False   # calculate G of the wiring stack if it wasn't saved during the simulation
latex_fonts = True

n_its = int(1E4)   # number of iterations for MC simulation
num_guesses = 100   # number of randomized initial guesses to try
analysis_dir = '/Users/angi/NIS/Bolotest_Analysis/'
# fn_comments = '_alpha0inf_1E4iteratinos_fitGconstantTTES_nobling_constrained'   # first go at getting errors, errors pulled from _errorcompare_master, incorrect sigma_I (mean not subtracted from and n-term in sigma_G)
# alim = [0,1]   # limits for fitting alpha
fn_comments = '_alpha0inf_1E4iteratinos_fitGconstantTTES_nobling'   # first go at getting errors, errors pulled from _errorcompare_master, incorrect sigma_I (mean not subtracted from and n-term in sigma_G)
alim = [0, np.inf]   # limits for fitting alpha
plot_comments = ''
vmax = 2E3   # quality plot color bar scaling
calc = 'mean'   # how to evaluate fit parameters from simluation data
bolo1b = True   # add bolo1 data to legacy prediction comparison plot


# initial guess for fitter
# p0_a0inf = np.array([0.74, 0.59, 1.29, 0.47, 1.94, 1.26]); sigmap0_a0inf = np.array([0.09, 0.19, 0.06, 0.58, 2.45, 0.11])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]; fit vals from alpha=[0,2] model
p0_a0inf_median = np.array([0.73, 0.61, 1.28, 0.20, 1.2, 1.25]);# sigmap0_a0inf_median = np.array([0.09, 0.19, 0.06, 0.58, 2.45, 0.11])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]; fit vals from alpha=[0,2] model
p0_a01 = np.array([0.8, 0.42, 1.33, 1., 1., 1.]); #sigmap0_a01 = np.array([0.03, 0.06, 0.03, 0.02, 0.03, 0.00])   #  fit vals from alpha=[0,1] model, 1E5 iterations
p0 = p0_a0inf_median;# sigma_p0 = sigmap0_a0inf_median

# choose GTES data 
# ydata_fitGexplicit = [13.95595194, 5.235218152, 8.182147122, 10.11727864, 17.47817158, 5.653424631, 15.94469664, 3.655108238]   # pW/K at 170 mK fitting for G explicitly, weighted average on most bolos (*) (second vals have extra bling); bolo 1b, 24*, 23*, 22, 21*, 20, 7*, 13* 
# sigma_fitGexplicit = [0.073477411, 0.01773206, 0.021512022, 0.04186006, 0.067666665, 0.014601341, 0.083450365, 0.013604177]   # fitting for G explicitly produces very small error bars
ydata_fitGexplicit_nobling = [13.95595194, 4.721712381, 7.89712938, 10.11727864, 17.22593561, 5.657104443, 15.94469664, 3.513915367]   # pW/K at 170 mK fitting for G explicitly, weighted average only on 7; bolo 1b, 24, 23, 22, 21, 20, 7*, 13 
sigma_fitGexplicit_nobling = [0.073477411, 0.034530085, 0.036798694, 0.04186006, 0.09953389, 0.015188074, 0.083450365, 0.01762426]
ydata = np.array(ydata_fitGexplicit_nobling); sigma = np.array(sigma_fitGexplicit_nobling)

bolos = np.array(['bolo 1b', 'bolo 24', 'bolo 23', 'bolo 22', 'bolo 21', 'bolo 20', 'bolo 7', 'bolo 13'])
bounds = [(0, np.inf), (0, np.inf), (0, np.inf), (alim[0], alim[1]), (alim[0], alim[1]), (alim[0], alim[1])]   # bounds for 6 fit parameters: G_U, G_W, G_I, alpha_U, alpha_W, alpha_I
plot_dir = analysis_dir + 'plots/layer_extraction_analysis/'
sim_file = analysis_dir + 'Analysis_Files/sim' + fn_comments + '.pkl'
csv_file = analysis_dir + 'Analysis_Files/sim' + fn_comments + '.csv'
data = [ydata, sigma] 

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
    sim_results = runsim_chisq(n_its, p0, data, bounds, plot_dir, show_yplots=show_plots, save_figs=save_figs, save_sim=save_sim, sim_file=sim_file, fn_comments=fn_comments)  


if quality_plots:   # plot G_x vs alpha_x parameter space with various fit results

    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
    sim_dataT = sim_dict['sim']; sim_data = sim_dataT.T   # simulation parameter values
    param_labels = ['G$_U$', 'G$_W$', 'G$_I$', '$\\alpha_U$', '$\\alpha_W$', '$\\alpha_I$']

    if np.isinf(alim[1]):   # quality plot title
        qp_title = '$\\boldsymbol{\\mathbf{\\alpha \\in [0,\infty)}}$'   # title for 1x3 quality plots
    else:
        qp_title = '$\\boldsymbol{\\mathbf{\\alpha \in [0,'+str(alim[1])+']}}$'   # title for 1x3 quality plots

    ### plot fit in 2D parameter space, take mean values of simulation
    results_mean = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_mean', title=qp_title+'\\textbf{ (Mean)}', vmax=vmax, calc='mean')
    params_mean, paramerrs_mean, kappas_mean, kappaerrs_mean, Gwire_mean, sigmaGwire_mean, chisq_mean = results_mean

    ### plot fit in 2D parameter space, take median values of simulation
    results_med = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_median', title=qp_title+'\\textbf{ (Median)}', vmax=vmax, calc='median')
    params_med, paramerrs_med, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_med

    ### pairwise correlation plots
    pairfig = pairwise(sim_data, param_labels, title=qp_title, save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments)

    ### analyze sub-populations of solutions 
    # zeroaW = np.where((sim_data[4] == 0))[0]
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - alphaW=0 solns}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments, indsop=zeroaW, oplotlabel='$\\alpha_W$=0')

    # aWlim = 1E-5; aUlim = 0.7   # limits to delineate two solution spaces
    # lowa = np.where((sim_data[4] < aWlim) & (sim_data[3] < aUlim))[0]
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W<}}$ '+str(aWlim)+' and $\\boldsymbol{\\mathbf{\\alpha_U<}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_overplotlowa', indsop=lowa, oplotlabel='low $\\alpha$')
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W<}}$ '+str(aWlim)+' and $\\boldsymbol{\\mathbf{\\alpha_U<}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_lowa', indstp=lowa)

    # higha = np.where((sim_data[4] > aWlim) | (sim_data[3] > aUlim))[0]   # hopefully this removes bimodal solutions
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W>}}$ '+str(aWlim)+' or $\\boldsymbol{\\mathbf{\\alpha_U>}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_overplothigha', indsop=higha, oplotlabel='high $\\alpha$')
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W>}}$ '+str(aWlim)+' or $\\boldsymbol{\\mathbf{\\alpha_U>}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_higha', indstp=higha)

    # print('\n\nAnalyzing only HIGH aW and aU solutions:')
    # results_higha = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_higha', title=qp_title+'\\textbf{, high aW and aU (Mean)}', vmax=vmax, calc='mean', spinds=higha)

    # print('\n\nAnalyzing only LOW aW and aU solutions:')
    # results_lowa = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_lowa', title=qp_title+'\\textbf{ low aW and aU (Mean)}', vmax=vmax, calc='mean', spinds=lowa)


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


if average_qp:   # check for scattering in full MC simulation vs single run

    funcgrid_U, funcgridmean_U = plot_meanqp(p0, data, n_its, 'U', plot_dir, savefigs=save_figs, fn_comments=fn_comments)
    funcgrid_W, funcgridmean_W = plot_meanqp(p0, data, n_its, 'W', plot_dir, savefigs=save_figs, fn_comments=fn_comments)
    funcgrid_I, funcgridmean_I = plot_meanqp(p0, data, n_its, 'I', plot_dir, savefigs=save_figs, fn_comments=fn_comments)


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

    f = 1   # fraction of diffuse reflections; 1 = Casimir limit
    mfpI_eff = l_eff(7, .4, f)   # um; reflection limited phonon mfp including spectral scattering (if f<1)
    gamma = mfpI_eff/mfpI_Wyb   # should be 1 if f=1 (Casimir limit)

    # reproduce Wyborne84 Fig 1
    dtest = np.ones(50); wtest = np.linspace(1,50)
    plt.figure(); plt.plot(wtest/dtest, (1-l_bl(wtest, dtest)/l_Casimir(wtest, dtest))*100)
    plt.xlabel('n')

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

    if np.isinf(alim[1]):
        plot_comments = '_unconstrained'
        title='Predictions from Model, $\\alpha \\in [0,\\infty)$'
    else:
        plot_comments = '_constrained'
        title='Predictions from Model, $\\alpha \\in [0,1]$'
    sim_dataT = sim_dict['sim']; sim_data = sim_dataT.T

    L = 220   # bolotest leg length, um
    boloGs = ydata; sigma_boloGs = sigma   # choose bolotest G's to compare
    wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20
    A_bolo = np.array([(7*4*.420+5*4*.160+3*4*.340+7*4*.350+7*4*.400), (7*1*.420+7*3*.340+5*.160+3*.340+7*.350+7*.400), (7*2*.420+7*2*.340+5*2*.160+3*2*.340+7*2*.350+7*2*.400), (7*3*.420+7*1*.340+5*3*.160+3*3*.340+7*3*.350+7*3*.400), (7*1*.420+7*3*.400+5*1*.160+3*1*.285+7*3*.370+7*1*.350), (7*4*.420+5*1*.160+wstack_width*3*.385+3*1*.285+7*1*.340), (7*3*.420+7*1*.400+5*3*.160+3*1*3.340+7*3*.350+7*1*.670+7*3*.400), (7*1*.420+7*3*.400+5*1*.160+3*1*.285+7*1*.350) ])   # bolotest areas
    AoL_bolo = A_bolo/L   # A/L for bolotest devices
    data1b=np.array([ydata[0], sigma[0]]) if bolo1b else []  # plot bolo1b data?

    predict_Glegacy(simresults_mean, data1b=data1b, save_figs=save_figs, title=title+' (Mean)', plot_comments=plot_comments+'_mean', fs=(7,7))
    predict_Glegacy(simresults_med, data1b=data1b, save_figs=save_figs, title=title+' (Median)', plot_comments=plot_comments+'_median')
    # (G_layer(simresults_mean, dI1, layer='I') + G_layer(simresults_mean, dI2, layer='I')) *lw/7 *220/ll   # not sure what this was for?

    # title="Predictions from Layer $\kappa$'s"
    # plot_comments = '_kappa'
    # predict_Glegacy(simresults_mean, data1b=data1b, save_figs=save_figs, estimator='kappa', title=title+' (Mean)', plot_comments=plot_comments)

    
if design_implications:   # making plots to illustrate TES and NIS design implications of this work

    ### plot G_TES and TFN as a function of substrate width
    # fit_params = p0_a0inf_median; sig_params = sigmap0_a0inf_median
    # fit = np.array([p0_a0inf_median, sigmap0_a0inf_median])
    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
    simresults_mean = np.array([np.mean(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)])
    simresults_med = np.array([np.median(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)])
    sim_results = simresults_mean if calc=='mean' else simresults_med

    lwidths = np.linspace(0.1, 100/7, num=100)   # um
    # llength = 220*np.ones_like(lwidths)   # um
    # dsub = .420*np.ones_like(lwidths)   # um    
    llength = 220   # um
    dsub = .420   # um
    Tbath = 0.170   # K
    def A_1b(lw, layer='wiring'):   # area of bolotest bolo with microstrip on four legs, i.e. bolo 1b
        if layer=='wiring':
            dsub = .420; dW1 = .160; dI1 = .350; dW2 = .340; dI2 = .400   # film thicknesses, um
        elif layer=='W1':
            dsub = .420; dW1 = .200; dI1 = 0; dW2 = 0; dI2 = 0   # film thicknesses, um
        elif layer=='bare':
            dsub = .340; dW1 = 0; dI1 = 0; dW2 = 0; dI2 = 0   # film thicknesses, um
        w1w, w2w = wlw(lw, fab='bolotest', layer=layer)
        return (lw*dsub + w1w*dW1 + w2w*dW2 + lw*dI1 +lw*dI2)*4   # area of four legs with microstrip, i.e. bolo 1b
    
    A_full = A_1b(lwidths, layer='wiring'); A_W1 = A_1b(lwidths, layer='W1'); A_bare = A_1b(lwidths, layer='bare')   # areas of different film stacks

    G_full, Gerr_full = Gfrommodel(sim_results, dsub, lwidths, llength, layer='total', fab='bolotest')
    G_U, Gerr_U = Gfrommodel(sim_results, dsub, lwidths, llength, layer='U', fab='bolotest')
    G_W1, Gerr_W1 = Gfrommodel(sim_results, dsub, lwidths, llength, layer='W1', fab='bolotest')
    G_Nb200 = G_U+G_W1; Gerr_Nb200 = Gerr_U+Gerr_W1
    G_bare, Gerr_bare = Gfrommodel(sim_results, .340, lwidths, llength, layer='U', fab='bolotest')   # bare substrate is thinner from etching steps

    NEP_full = TFNEP(Tbath, G_full*1E-12)*1E18; NEPerr_full = TFNEP(Tbath, Gerr_full*1E-12)*1E18   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK
    NEP_W1 = TFNEP(Tbath, G_Nb200*1E-12)*1E18; NEPerr_W1 = TFNEP(Tbath, Gerr_Nb200*1E-12)*1E18   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK
    NEP_bare = TFNEP(Tbath, G_bare*1E-12)*1E18; NEPerr_bare = TFNEP(Tbath, Gerr_bare*1E-12)*1E18   # aW / rtHz; Kenyan 2006 measured 1E-17 for a TES with comparable G at 170 mK
    Glims = np.array([0, 1.1*np.nanmax([G_full, G_W1, G_bare])])
    NEPlims = TFNEP(Tbath, Glims*1E-12)*1E18
    
    # predicted G and NEP vs substrate width
    fig, ax1 = plt.subplots() 
    ax1.plot(lwidths, G_full, color='rebeccapurple', label='G$_\\text{TES}$, Microstrip', alpha=0.8) 
    plt.fill_between(lwidths, G_full-Gerr_full, G_full+Gerr_full, facecolor="mediumpurple", alpha=0.2)   # error
    ax1.plot(lwidths, G_Nb200, color='green', label='G$_\\text{TES}$, 200nm Nb', alpha=0.8) 
    plt.fill_between(lwidths, G_Nb200-Gerr_Nb200, G_Nb200+Gerr_Nb200, facecolor="limegreen", alpha=0.2)   # error
    ax1.plot(lwidths, G_bare, color='royalblue', label='G$_\\text{TES}$, Bare', alpha=0.8)
    plt.fill_between(lwidths, G_bare-Gerr_bare, G_bare+Gerr_bare, facecolor="cornflowerblue", alpha=0.2)   # error
    ax1.set_xlabel('Substrate Width [$\mu$m]') 
    ax1.set_ylabel('G$_\\text{TES}$ [pW/K]') 
    ax1.set_ylim(ymin=Glims[0], ymax=Glims[1]) 
    ax2 = ax1.twinx() 
    ax2.plot(lwidths, NEP_full, '--', color='rebeccapurple', label='NEP')   # this varies as G^1/2
    ax2.plot(lwidths, NEP_W1, '--', color='green', label='NEP')   # this varies as G^1/2
    ax2.plot(lwidths, NEP_bare, '--', color='royalblue', label='NEP')   # this varies as G^1/2
    ax2.set_ylim(ymin=NEPlims[0], ymax=NEPlims[1]) 
    ax2.set_ylabel('Thermal Fluctuation NEP [aW/$\sqrt{Hz}$]')     
    ax2.set_xlim(0, np.max(lwidths))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize='12', ncol=2)
    if save_figs: plt.savefig(plot_dir + 'design_implications_v2.png', dpi=300) 
   
    # predicted G and NEP vs leg A/L
    fig, ax1 = plt.subplots() 
    ax1.plot(A_full/llength, G_full, color='rebeccapurple', label='G$_\\text{TES}$, Microstrip', alpha=0.8) 
    plt.fill_between(A_full/llength, G_full-Gerr_full, G_full+Gerr_full, facecolor="mediumpurple", alpha=0.2)   # error
    ax1.plot(A_W1/llength, G_Nb200, color='green', label='G$_\\text{TES}$, 200nm Nb', alpha=0.8) 
    plt.fill_between(A_W1/llength, G_Nb200-Gerr_Nb200, G_Nb200+Gerr_Nb200, facecolor="limegreen", alpha=0.2)   # error
    ax1.plot(A_bare/llength, G_bare, color='royalblue', label='G$_\\text{TES}$, Bare', alpha=0.8)
    plt.fill_between(A_bare/llength, G_bare-Gerr_bare, G_bare+Gerr_bare, facecolor="cornflowerblue", alpha=0.2)   # error
    ax1.set_xlabel('TES Leg A/L [$\mu$m]') 
    ax1.set_ylabel('G$_\\text{TES}$ [pW/K]') 
    ax1.set_ylim(ymin=Glims[0], ymax=Glims[1]) 
    ax2 = ax1.twinx() 
    ax2.plot(A_full/llength, NEP_full, '--', color='rebeccapurple', label='NEP')   # this varies as G^1/2
    ax2.plot(A_W1/llength, NEP_W1, '--', color='green', label='NEP')   # this varies as G^1/2
    ax2.plot(A_bare/llength, NEP_bare, '--', color='royalblue', label='NEP')   # this varies as G^1/2
    ax2.set_ylim(ymin=NEPlims[0], ymax=NEPlims[1]) 
    ax2.set_ylabel('Thermal Fluctuation NEP [aW/$\sqrt{Hz}$]')     
    ax2.set_xlim(0, np.nanmax(A_full/llength))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize='12', ncol=2)
    if save_figs: plt.savefig(plot_dir + 'design_implications_AL.png', dpi=300) 

    # looking for crossover dsub value where G_micro = G_sub 
    # this will be constant with legnth but not constant with width since w_W1/W2 are constant
    # at lw=10 um, the crossover thickness is 1.17  um
    # at lw=7 um, the crossover thickness is 1.14  um
    # at lw=5 um, the crossover thickness is 1.21  um
    # at lw=41.5 um, the crossover thickness is 1.02  um

    lw_test = 20; ll_test = 20   # um; um
    fab='legacy'
    # dsubs = np.linspace(0, 2, int(1E4))  # um
    # G_micros = Gfrommodel(p0, dsubs, lw_test, ll_test, layer='wiring', fab='bolotest')
    # G_Us = Gfrommodel(p0, dsubs, lw_test, ll_test, layer='U', fab='bolotest')
    # Gratio = G_Us/G_micros
    def xover_finder(dsub, fit, lw, ll, fab=fab):   # when G_U/G_micro = 1, returns 0
        G_micros = Gfrommodel(fit, dsub, lw, ll, layer='wiring', fab=fab)[0]
        G_Us = Gfrommodel(fit, dsub, lw, ll, layer='U', fab=fab)[0]
        return G_Us/G_micros - 1

    # ll_test = 1   # um, shouldn't matter since ratio is length independent
    # lwidths = np.arange(0, 50, 10)
    # xover_dsub = fsolve(xover_finder, 1, args=(np.stack(([p0]*len(lwidths)),axis=0), lwidths, np.ones_like(lwidths)*ll_test))[0]   # um
    # xover_dsub = fsolve(xover_finder, 1, args=(np.stack(([p0]*len(lwidths)),axis=0)[0], lwidths[0], (np.ones_like(lwidths)*ll_test)[0]))[0]   # um
    # fsolve(xover_finder, 1, args=(np.stack(([p0]*len(lwidths)),axis=0), lwidths, (np.ones_like(lwidths)*ll_test)))[0]
    xover_dsub = fsolve(xover_finder, ll_test, args=(sim_results, lw_test, ll_test))[0]


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

    # chisq_fit = calc_chisq(ydata, Gbolos(sim_results[0]))
    # print('Chi-squared value for the fit: ', round(chisq_fit, 3))
    
    # parameters = np.array(['G_U', 'G_W', 'G_I', 'alpha_U', 'alpha_W', 'alpha_I'])
    # # look at histogram of fit values for each parameter
    # for cc, col in enumerate(sim_data.T):
    #     plt.figure()
    #     plt.hist(col, bins=20)
    #     plt.title(parameters[cc]+' Values for alpha=[0,inf)')
    #     plt.yscale('log')
        
    # plt.figure()
    # plt.hist(Gwires, bins=20)
    # plt.title('G_wire Values for alpha=[0,inf)')
    # plt.yscale('log')

if scrap:

    # estimate d dependence in l_eff for purely diffusive limit
    def monoExp(x, m, t, b):
        return m * np.exp(t * x) + b
    def expdecay(x, m, t, b):
        return m * np.exp(-t * x) + b
    def plaw(x, a, b, c):
        return a*x**b + c

    ntest = np.linspace(1,30)   # range of n we care about, bolotest n=17.5
    Iterms = ntest**3*I_mfp(1/ntest)+I_mfp(ntest)

    # polynomial fits
    pparams3 = np.polyfit(ntest, Iterms, 3)
    pparams2 = np.polyfit(ntest, Iterms, 2)
    pparams1 = np.polyfit(ntest, Iterms, 1)

    # power law fit
    yp1 = np.log(Iterms); xp1 = np.log(ntest)   # transform data
    tparams1 = np.polyfit(xp1, yp1, 1)
    
    # exponential fit
    yp2 = np.log(Iterms); xp2 = ntest   # transform data
    tparams2 = np.polyfit(xp2, yp2, 1)

    # power law with offset
    from scipy.optimize import curve_fit
    p0test = (0.9, 1.3, -0.5) 
    ploparams, cv = curve_fit(plaw, ntest, Iterms, p0test)

    # resdiuals 
    plt.figure()
    plt.plot(ntest, (pparams3[0]*ntest**3 + pparams3[1]*ntest**2 + pparams3[2]*ntest + pparams3[3]-Iterms)/Iterms*100, '.', label='3 deg polynomial')
    plt.plot(ntest, (pparams2[0]*ntest**2 + pparams2[1]*ntest + pparams2[2]-Iterms)/Iterms*100, '.', label='2 deg polynomial')
    # plt.plot(ntest, (pparams1[0]*ntest + pparams1[1]-Iterms)/Iterms*100, '.', label='1 deg polynomial')   # this fit is the worst of the power law fits
    plt.plot(ntest, (np.exp(tparams1[1])*ntest**tparams1[0]-Iterms)/Iterms*100, '.', label=str(round(np.exp(tparams1[1]),2)) + 'n^'+str(round(tparams1[0], 2)))
    plt.plot(ntest, (plaw(ntest, ploparams[0], ploparams[1], ploparams[2])-Iterms)/Iterms*100, '.', label=str(round(ploparams[0],2)) + 'n^' + str(round(ploparams[1], 2))+str(round(ploparams[2], 2)))
    # plt.plot(ntest, (np.exp(tparams2[1]*ntest*tparams2[0])-Iterms)/Iterms*100, '.', label='exponential fit, beta='+str(round(tparams1[0], 2)))   # exponential fit is the most worst of them all
    plt.xlabel('n')
    plt.ylabel('Residuals [%]')
    plt.legend()
    plt.ylim(-4,4)
    plt.xlim(1,30)
    plt.vlines(17.5, -4, 4, color='k')
    plt.hlines(0, 1, 30, linestyles='--', color='k')
    plt.title('n$^3$I(1/n) + I(n)')

    from lmfit import Model
    fmodel = Model(Gbolos)
    result = fmodel.fit(y, x=x, a=14, b=3.9, mo=0.8, q=0.002)
    sim_result = minimize(chisq_val, p0, args=[ydata, sigma], bounds=bounds)
    curve_fit(self.powerlaw_fit_func, temperatures, powerAtRns[index], p0=init_guess, sigma=sigma[index], absolute_sigma=True) 

if bimodal_solns:

    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
    sim_dataT = sim_dict['sim']; sim_data = sim_dataT.T   # simulation parameter values
    param_labels = ['G$_U$', 'G$_W$', 'G$_I$', '$\\alpha_U$', '$\\alpha_W$', '$\\alpha_I$']

    if np.isinf(alim[1]):   # quality plot title
        qp_title = '$\\boldsymbol{\\mathbf{\\alpha \\in [0,\infty)}}$'   # title for 1x3 quality plots
    else:
        qp_title = '$\\boldsymbol{\\mathbf{\\alpha \in [0,'+str(alim[1])+']}}$'   # title for 1x3 quality plots

    ### pairwise correlation plots
    pairfig = pairwise(sim_data, param_labels, title=qp_title, save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments)

    # ### analyze sub-populations of solutions 
    aWlim = 1E-5; aUlim = 0.7   # limits to delineate two solution spaces
    lowa = np.where((sim_data[4] < aWlim) & (sim_data[3] < aUlim))[0]
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W<}}$ '+str(aWlim)+' and $\\boldsymbol{\\mathbf{\\alpha_U<}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_overplotlowa', indsop=lowa, oplotlabel='low $\\alpha$')
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W<}}$ '+str(aWlim)+' and $\\boldsymbol{\\mathbf{\\alpha_U<}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_lowa', indstp=lowa)

    higha = np.where((sim_data[4] > aWlim) | (sim_data[3] > aUlim))[0]   # hopefully this removes bimodal solutions
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W>}}$ '+str(aWlim)+' or $\\boldsymbol{\\mathbf{\\alpha_U>}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_overplothigha', indsop=higha, oplotlabel='high $\\alpha$')
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W>}}$ '+str(aWlim)+' or $\\boldsymbol{\\mathbf{\\alpha_U>}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_higha', indstp=higha)

    print('\n\nAnalyzing only HIGH aW and aU solutions:')
    results_higha = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_higha', title=qp_title+'\\textbf{, high aW and aU (Mean)}', vmax=vmax, calc='mean', spinds=higha, plot=False)
    params_higha, paramerrs_higha, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_higha
    fit_higha = np.array([params_higha, paramerrs_higha])

    print('\n\nAnalyzing only LOW aW and aU solutions:')
    results_lowa = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_lowa', title=qp_title+'\\textbf{ low aW and aU (Mean)}', vmax=vmax, calc='mean', spinds=lowa, plot=False)
    params_lowa, paramerrs_lowa, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_lowa
    fit_lowa = np.array([params_lowa, paramerrs_lowa])

    ### compare legacy predictions
    data1b=np.array([ydata[0], sigma[0]]) if bolo1b else []  # plot bolo1b data?

    predict_Glegacy(fit_higha, data1b=data1b, save_figs=save_figs, title=qp_title+' (High Alpha)', plot_comments=plot_comments+'_higha', fs=(7,7))
    predict_Glegacy(fit_lowa, data1b=data1b, save_figs=save_figs, title=qp_title+' (Low Alpha)', plot_comments=plot_comments+'_lowa')

    aWlim = 1.5E-4; GUlim = 0.63   # limits to delineate two solution spaces
    lowGU = np.where((sim_data[4] < aWlim) & (sim_data[0] < GUlim))[0]
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{G_U<}}$ '+str(GUlim)+' pW/K Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_overplotlowGU', indsop=lowGU, oplotlabel='low GU')
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W<}}$ '+str(aWlim)+' and $\\boldsymbol{\\mathbf{\\alpha_U<}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_lowGU', indstp=lowGU)

    highGU = np.where((sim_data[4] > aWlim) | (sim_data[0] > GUlim))[0]
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{G_U>}}$ '+str(GUlim)+' pW/K Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_overplotlowGU', indsop=highGU, oplotlabel='high GU')
    # pairfig = pairwise(sim_data, param_labels, title=qp_title+'\\textbf{ - $\\boldsymbol{\\mathbf{\\alpha_W>}}$ '+str(aWlim)+' or $\\boldsymbol{\\mathbf{\\alpha_U>}}$ '+str(aUlim)+' Solutions}', save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments+'_higha', indstp=higha)

    print('\n\nAnalyzing only HIGH GU solutions:')
    results_highGU = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_highGU', title=qp_title+'\\textbf{, High GU (Mean)}', vmax=vmax, calc='mean', spinds=highGU, plot=False)
    params_highGU, paramerrs_highGU, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_highGU
    fit_highGU = np.array([params_highGU, paramerrs_highGU])

    print('\n\nAnalyzing only LOW GU solutions:')
    results_lowGU = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_lowGU', title=qp_title+'\\textbf{, Low GU (Mean)}', vmax=vmax, calc='mean', spinds=lowGU, plot=False)
    params_lowGU, paramerrs_lowGU, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_lowGU
    fit_lowGU = np.array([params_lowGU, paramerrs_lowGU])

    # compare legacy predictions
    predict_Glegacy(fit_highGU, data1b=data1b, save_figs=save_figs, title=qp_title+' (High GU)', plot_comments=plot_comments+'_highGU', fs=(7,7))
    predict_Glegacy(fit_lowGU, data1b=data1b, save_figs=save_figs, title=qp_title+' (Low GU)', plot_comments=plot_comments+'_lowGU')

if compare_modelanddata:

    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
    sim_dataT = sim_dict['sim']; sim_data = sim_dataT.T   # simulation parameter values
    simresults_mean = np.array([np.mean(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)])
    simresults_med = np.array([np.median(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)])

    if np.isinf(alim[1]):   # quality plot title
        title = '$\\boldsymbol{\\mathbf{\\alpha \\in [0,\infty)}}$'   # plot title
    else:
        title = '$\\boldsymbol{\\mathbf{\\alpha \in [0,'+str(alim[1])+']}}$'   # plot title

    plot_modelvdata(simresults_mean, data, title=title+' (Mean)')

    # check subsections of solutions
    aWlim = 1.5E-4; GUlim = 0.63   # limits to delineate two solution spaces
    lowGU = np.where((sim_data[4] < aWlim) & (sim_data[0] < GUlim))[0]
    highGU = np.where((sim_data[4] > aWlim) | (sim_data[0] > GUlim))[0]

    results_highGU = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_highgu', title=title+'\\textbf{, high aW and aU (Mean)}', vmax=vmax, calc='mean', spinds=highGU, plot=False)
    params_highGU, paramerrs_highGU, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_highGU
    fit_highGU = np.array([params_highGU, paramerrs_highGU])

    # print('\n\nAnalyzing only LOW aW and aU solutions:')
    results_lowGU = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_lowGU', title=title+'\\textbf{ low aW and aU (Mean)}', vmax=vmax, calc='mean', spinds=lowGU, plot=False)
    params_lowGU, paramerrs_lowGU, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_lowGU
    fit_lowGU = np.array([params_lowGU, paramerrs_lowGU])

    plot_modelvdata(fit_lowGU, data, title=title+', Low GU Solns')
    plot_modelvdata(fit_highGU, data, title=title+', High GU Solns')


plt.show()