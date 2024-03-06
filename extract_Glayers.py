"""
Script for measuring individual layer contributions to the total thermal conductivity of a TES. 
Data from 'bolotest' measurements run in 2018. 
G_total and error is from TES power law fit measured at Tc and scaled to 170 mK, assuming dP/dT = G = n*k*Tc^(n-1).
Layers: U = SiN + SiO substrate, W = Nb wiring layers (<=2 layers per leg), I = SiN insulating layers (<=2 layers per leg)

aharkehosemann@gmail.com
2022/01


UPDATES :

2024/02/28 : added ability to bootstrap thicknesses within layer-specific error bars
Adding two-layer functionality in which substrate and insulating nitride layers are treated as the same layer, and d0_U = 400 nm instead of 420 in three-layer model
note: legacy geometry has not been changed, though these nitride layers may also be thinner than predicted?

2024/02/29 : changed substrate d0 from 420 nm to 400 nm for uniformity. Still in the middle of adding two-layer model. 

2024/03/06 : added G suppression of legs B, E, and G substrates due to surface roughness. 5% suppression seems to be a better fit than 10%.  


TODO: two layer model where all nitride is treated the same?, revisit error bars on layer widths, add error of d in chi-sq calc?, 
handle highly variable texture, joel is interested in legacy predictions with fixed normalized residuals
"""
from bolotest_routines import *
from scipy.optimize import fsolve
import csv 

### User Switches
# analysis
run_sim = False   # run MC simulation for fitting model
quality_plots = False   # results on G_x vs alpha_x parameter space for each layer
pairwise_plots = True   # histogram and correlations of simulated fit parameters
compare_modelanddata = True   # plot model predictions and bolotest data
compare_legacy = True   # compare with NIST sub-mm bolo legacy data
lit_compare = False   # compare measured conductivities with values from literature
design_implications = False   # NEP predictions from resulting model
analyze_vlengthdata = False   # look at bolotest data vs leg length
manual_params = False   # pick parameters manually and compare data
scrap = False

# options
model = 'three-layer'   # two- or three-layer model?
constrained = False   # use constrained model results
n_its = int(1E3)   # number of iterations in MC simulation
vmax = 1E4   # quality plot color bar scaling
calc = 'Median'   # how to evaluate fit parameters from simluation data - options are 'Mean' and 'Median'
qplim = [-1,2]   # x- and y-axis limits for quality plot
plot_bolo1b = True   # add bolo1 data to legacy prediction comparison plot, might turn off for paper figures
show_simGdata = False   # show simulated y-data plots during MC simulation

# save results
save_figs = True   # save figures 
save_sim = True   # save simulation data
save_csv = True   # save csv file of results

# where to save results
analysis_dir = '/Users/angi/NIS/Bolotest_Analysis/'
# fn_comments = '_postFIB_varyd'; vary_thickness = True; # vary film thickness, layer-specific error bars
# fn_comments = '_postFIB_varyd_originald0s'; vary_thickness = False; # don't vary film thickness, original d estimates
# fn_comments = '_postFIB_originald0s'; vary_thickness = False; # don't vary film thickness, original d estimates
# fn_comments = '_twolayermodel'; vary_thickness = False; # vary film thickness, layer-specific error bars
# fn_comments = '_twolayermodel_varythickness'; vary_thickness = True; # vary film thickness, layer-specific error bars
# fn_comments = '_twolayermodel_varythickness_suppressGU'; vary_thickness = True; # vary film thickness, suppress G of roughened substrate layers
# fn_comments = '_threelayermodel_varythickness_suppressGU'; vary_thickness = True; # vary film thickness, suppress G of roughened substrate layers
# fn_comments = '_threelayermodel_varythickness_increasedU150nm'; vary_thickness = True; # vary film thickness, add extra material to leg B
# fn_comments = '_twolayermodel_varythickness_increasedU150nm'; vary_thickness = True; # vary film thickness, add extra material to leg B
# fn_comments = '_twolayermodel_varythickness_increasedU150nm_G0unconstrained'; vary_thickness = True; # vary film thickness, add extra material to leg B, allow -G0
# fn_comments = '_threelayermodel_varythickness_suppresstrenched10pc'; vary_thickness = True; # vary film thickness, treat substrate on legs B, E, and G as trenched
fn_comments = '_threelayermodel_varythickness_suppresstrenched_5pc'; vary_thickness = True; # vary film thickness, treat substrate on legs B, E, and G as trenched
# sigmaG_frac = 0.; 

### layer thicknesses values; default from 2 rounds of FIB measurements: layer_ds = np.array([0.372, 0.312, 0.199, 0.181, 0.162, 0.418, 0.298, 0.596, 0.354, 0.314, 0.302])
# dS_ABD = 0.372; dS_CF = 0.312; dS_E1 = 0.108; dS_E2 = 0.321; dS_G = 0.181   # [um] substrate thickness for different legs, originally 420, 400, 420, 420, 340
dS_ABD = 0.384; dS_CF = 0.321; dS_E1 = 0.164; dS_E2 = 0.345; dS_G = 0.235   # [um] substrate thickness for different legs, originally 420, 400, 420, 420, 340
dW1_ABD = 0.162; dW1_E = 0.418   # [um] W1 thickness for different legs, originally 160, 100+285
dI1_ABC = 0.298; dI_DF = 0.596   # [um] I1 thickness for different legs, originally 350, 270+400
# dI1_ABC = 0.258; dI_DF = 0.610   # [um] I1 thickness for different legs, originally 350, 270+400
dW2_AC = 0.354; dW2_BE = 0.314   # [um] W2 thickness for different legs, originally 340, 285
dI2_AC = 0.302   # [um] I2 thickness, originally 400
# dI2_AC = 0.252   # [um] I2 thickness, originally 400

# ### layer thicknesses values; default from 2 rounds of FIB measurements: layer_ds = np.array([0.372, 0.312, 0.199, 0.181, 0.162, 0.418, 0.298, 0.596, 0.354, 0.314, 0.302])
# dS_ABD = 0.420; dS_CF = 0.400; dS_E1 = 0.420; dS_E2 = 0.420; dS_G = 0.340   # [um] substrate thickness for different legs, originally 420, 400, 420, 420, 340
# dW1_ABD = 0.160; dW1_E = 0.385   # [um] W1 thickness for different legs, originally 160, 100+285
# dI1_ABC = 0.350; dI_DF = 0.670   # [um] I1 thickness for different legs, originally 350, 270+400
# dW2_AC = 0.340; dW2_BE = 0.285   # [um] W2 thickness for different legs, originally 340, 285
# dI2_AC = 0.400   # [um] I2 thickness, originally 400

# error bars
# dSABD_err = 0.032; dSCF_err = 0.012; dSE1_err = 0.052; dSE2_err = 0.024; dSG_err = 0.048   # [um] substrate thickness error bars
dSABD_err = 0.016; dSCF_err = 0.012; dSE1_err = 0.052; dSE2_err = 0.024; dSG_err = 0.048   # [um] substrate thickness error bars
dW1ABD_err = 0.008; dW1E_err = 0.00   # [um] W1 thickness error bars, originally 160, 100 nm
dI1ABC_err = 0.040; dIDF_err = 0.018   # [um] I1 thickness error bars, originally 350, 270 nm
dW2AC_err = 0.013; dW2BE_err = 0.012   # [um] W2 thickness error bars, originally 340, 285 nm
dI2AC_err = 0.047   # [um] I2 thickness, originally 400

dS_E = (4*dS_E1 + 3*dS_E2)/7; dSE_err = (4*dSE1_err + 3*dSE2_err)/7  # width-weighted thickness, S layer with step in height
layer_d0 = np.array([dS_ABD, dS_CF, dS_E, dS_G, dW1_ABD,  dW1_E, dI1_ABC, dI_DF, dW2_AC, dW2_BE, dI2_AC])
derrs = np.array([dSABD_err, dSCF_err, dSE_err, dSG_err, dW1ABD_err,  dW1E_err, dI1ABC_err, dIDF_err, dW2AC_err, dW2BE_err, dI2AC_err])
if vary_thickness==False: derrs = np.zeros_like(layer_d0)  # turn off errors if not varying thickness (shouldn't matter but for good measure)

# initial guess and bounds for fitter
alim = [-1, 1] if constrained else [-np.inf, np.inf]   # limit alpha to [-1, 1] if constraining fit
if model=='three-layer':
    p0 = np.array([1., 0.75, 1., 0.5, 0., 1.])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]
    # bounds = [(0, np.inf), (0, np.inf), (0, np.inf), (alim[0], alim[1]), (alim[0], alim[1]), (alim[0], alim[1])]   # bounds for 6 fit parameters: G_U, G_W, G_I, alpha_U, alpha_W, alpha_I
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (alim[0], alim[1]), (alim[0], alim[1]), (alim[0], alim[1])]   # bounds for 6 fit parameters: G_U, G_W, G_I, alpha_U, alpha_W, alpha_I
elif model=='two-layer':
    p0 = np.array([1, 0.5, 0.5, 0.5])   # U, W [pW/K], alpha_U, alpha_W [unitless]
    # bounds = [(0, np.inf), (0, np.inf), (alim[0], alim[1]), (alim[0], alim[1])]   # bounds for 4 fit parameters: G_U, G_W, G_I, alpha_U, alpha_W, alpha_I
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (alim[0], alim[1]), (alim[0], alim[1])]   # bounds for 4 fit parameters: G_U, G_W, G_I, alpha_U, alpha_W, alpha_I

# G_TES data 
ydata_lessbling = np.array([13.95595194, 4.721712381, 7.89712938, 10.11727864, 17.22593561, 5.657104443, 15.94469664, 3.513915367])   # pW/K at 170 mK fitting for G explicitly, weighted average only on 7; bolo 1b, 24, 23, 22, 21, 20, 7*, 13 
sigma_lessbling = np.array([0.073477411, 0.034530085, 0.036798694, 0.04186006, 0.09953389, 0.015188074, 0.083450365, 0.01762426])
ydata = ydata_lessbling; sigma = sigma_lessbling
# sigma_percentG = sigmaG_frac * ydata_lessbling

bolos = np.array(['bolo 1b', 'bolo 24', 'bolo 23', 'bolo 22', 'bolo 21', 'bolo 20', 'bolo 7', 'bolo 13'])
plot_dir = analysis_dir + 'Plots/layer_extraction_analysis/'; sim_file = analysis_dir + 'Analysis_Files/sim' + fn_comments + '.pkl'; csv_file = analysis_dir + 'Analysis_Files/sim' + fn_comments + '_results.csv'
data = [ydata, sigma] 

# bolos 1a-f vary leg length, all four legs have full microstrip
ydatavl_lb = np.array([22.17908389, 13.95595194, 10.21776418, 8.611287109, 7.207435165, np.nan]); sigmavl_lb = np.array([0.229136979, 0.073477411, 0.044379343, 0.027206631, 0.024171663, np.nan])   # pW/K, bolos with minimal bling
ydatavl_mb = np.array([22.19872947, np.nan, 10.64604145, 8.316849305, 7.560603448, 6.896700236]); sigmavl_mb = np.array([0.210591249, np.nan, 0.065518258, 0.060347632, 0.051737016, 0.039851469])   # pW/K, bolos with extra bling
ll_vl = np.array([120., 220., 320., 420., 520., 620.])*1E-3   # mm
llvl_all = np.append(ll_vl, ll_vl); ydatavl_all = np.append(ydatavl_lb, ydatavl_mb); sigmavl_all = np.append(sigmavl_lb, sigmavl_mb)
vlength_data = np.stack([ydatavl_all, sigmavl_all, llvl_all*1E3])

# for plotting
if model=='three-layer':
    param_labels = ['G$_U$', 'G$_W$', 'G$_I$', '$\\alpha_U$', '$\\alpha_W$', '$\\alpha_I$']  
elif model=='two-layer':
    param_labels = ['G$_U$', 'G$_W$', '$\\alpha_U$', '$\\alpha_W$']  
a0str = '(-\infty' if np.isinf(alim[0]) else '['+str(alim[0]); a1str = '\infty)' if np.isinf(alim[1]) else str(alim[1])+']'
plot_title = '$\\boldsymbol{\\mathbf{\\alpha \in '+a0str+','+a1str+'}}$'   

# use LaTeX fonts for paper plots
plt.rc('text', usetex=True); plt.rc('font', family='serif'); plt.rc('font', size=15); plt.rc('font', weight='normal')
plt.rcParams['text.latex.preamble']="\\usepackage{amsmath}"; plt.rcParams['xtick.major.size'] = 5; plt.rcParams['xtick.minor.visible'] = False    
plt.rcParams['ytick.major.size'] = 5; plt.rcParams['ytick.minor.visible'] = False

### Execute Analysis
if run_sim:   # run simulation 
    sim_dict = runsim_chisq(n_its, p0, data, bounds, plot_dir, show_simGdata=show_simGdata, save_figs=save_figs, save_sim=save_sim, sim_file=sim_file, 
                            fn_comments=fn_comments, calc=calc, vary_thickness=vary_thickness, derrs=derrs, layer_d0=layer_d0, model=model)  
else:   # load simulation data
    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
sim_data = sim_dict['sim']

if quality_plots:   # plot G_x vs alpha_x parameter space with various fit results
    ### plot fit in 2D parameter space, take mean values of simulation
    results_mean = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_mean', title=plot_title+'\\textbf{ (Mean)}', vmax=vmax, calc='Mean', qplim=qplim, layer_ds=layer_d0, model=model)
    params_mean, paramerrs_mean, kappas_mean, kappaerrs_mean, Gwire_mean, sigmaGwire_mean, chisq_mean = results_mean

    ### plot fit in 2D parameter space, take median values of simulation
    results_med = qualityplots(data, sim_dict, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments+'_median', title=plot_title+'\\textbf{ (Median)}', vmax=vmax, calc='Median', qplim=qplim, layer_ds=layer_d0, model=model)
    params_med, paramerrs_med, kappas_med, kappaerrs_med, Gwire_med, sigmaGwire_med, chisq_med = results_med

    if save_csv:   # save model results to CSV file
        if model=='three-layer':
            vals_mean = np.array([params_mean[0], params_mean[1], params_mean[2], params_mean[3], params_mean[4], params_mean[5], kappas_mean[0], kappas_mean[1], kappas_mean[2], Gwire_mean, chisq_mean])
            vals_med = np.array([params_med[0], params_med[1], params_med[2], params_med[3], params_med[4], params_med[5], kappas_med[0], kappas_med[1], kappas_med[2], Gwire_med, chisq_med])
            vals_err = np.array([paramerrs_mean[0], paramerrs_mean[1], paramerrs_mean[2], paramerrs_mean[3], paramerrs_mean[4], paramerrs_mean[5], kappaerrs_mean[0], kappaerrs_mean[1], kappaerrs_mean[2], sigmaGwire_mean, ''])   # should be the same for mean and median
            csv_params = np.array(['GU (pW/K)', 'GW (pW/K)', 'GI (pW/K)', 'alphaU', 'alphaW', 'alphaI', 'kappaU (pW/K/um)', 'kappaW (pW/K/um)', 'kappaI (pW/K/um)', 'Gwire (pW/K)', 'Chi-sq val'])
        elif model=='two-layer':
            vals_mean = np.array([params_mean[0], params_mean[1], params_mean[2], params_mean[3], kappas_mean[0], kappas_mean[1], Gwire_mean, chisq_mean])
            vals_med = np.array([params_med[0], params_med[1], params_med[2], params_med[3], kappas_med[0], kappas_med[1], Gwire_med, chisq_med])
            vals_err = np.array([paramerrs_mean[0], paramerrs_mean[1], paramerrs_mean[2], paramerrs_mean[3], kappaerrs_mean[0], kappaerrs_mean[1], sigmaGwire_mean, ''])   # should be the same for mean and median
            csv_params = np.array(['GU (pW/K)', 'GW (pW/K)', 'alphaU', 'alphaW', 'kappaU (pW/K/um)', 'kappaW (pW/K/um)', 'Gwire (pW/K)', 'Chi-sq val'])

        # write CSV     
        fields = np.array(['Parameter', 'Mean', 'Median', 'Error'])  
        rows = [[csv_params[rr], vals_mean[rr], vals_med[rr], vals_err[rr]] for rr in np.arange(len(csv_params))]
        with open(csv_file, 'w') as csvfile:  
            csvwriter = csv.writer(csvfile)  # csv writer object  
            csvwriter.writerow(fields)  
            csvwriter.writerows(rows)


if pairwise_plots:   ### pairwise correlation plots

    pairfig = pairwise(sim_data, param_labels, title=plot_title, save_figs=save_figs, plot_dir=plot_dir, fn_comments=fn_comments)


if compare_modelanddata:

    title = plot_title+'$\\textbf{ Predictions}$'   
    plot_modelvdata(sim_data, data, title=title+'$\\textbf{ ('+calc+')}$', layer_ds=layer_d0, pred_wfit=False, calc=calc, model=model, 
                    save_figs=save_figs, plot_comments=fn_comments+'_'+calc, plot_dir=plot_dir)


if compare_legacy:   # compare G predictions with NIST legacy data
 
    title = plot_title+'$\\textbf{ Predictions}$'   

    L = 220   # bolotest leg length, um
    boloGs = ydata; sigma_boloGs = sigma   # choose bolotest G's to compare
    AoL_bolo = bolotest_AoL(L=L, layer_ds=layer_d0)   # A/L for bolotest devices
    data1b = np.array([ydata[0], sigma[0]]) if plot_bolo1b else []  # plot bolo1b data?

    # compare with legacy data
    predict_Glegacy(sim_data, data1b=data1b, pred_wfit=False, calc=calc, save_figs=save_figs, title=title+'$\\textbf{ ('+calc+')}$', plot_comments=fn_comments+'_'+calc, fs=(7,7), model=model)
    predict_Glegacy(sim_data, data1b=data1b, pred_wfit=False, calc='Median', save_figs=save_figs, title=title+'$\\textbf{ (Median), }\\mathbf{\\beta=0.8}$', plot_comments=fn_comments+'_median_beta0p8', fs=(7,7), Lscale=0.8, model=model)
    # predict_Glegacy(sim_data, data1b=data1b, pred_wfit=False, calc='Mean', save_figs=save_figs, title=title+'$\\textbf{ (Mean)}$', plot_comments=fn_comments+'_mean', fs=(7,7))
    # predict_Glegacy(sim_data, data1b=data1b, pred_wfit=False, calc='Mean', save_figs=save_figs, title=title+'$\\textbf{ (Mean), }\\mathbf{\\beta=0.8}$', plot_comments=fn_comments+'_mean_beta0p8', fs=(7,7), Lscale=0.8)
    
    title = '$\\textbf{ Legacy Data - }$'   
    lAoLscale = 1.8
    plot_comments='_scaled1p8'
    # plot legacy data by itself, scale A/L?
    # plot_Glegacy(data1b=data1b, save_figs=save_figs, title=title+'$\\textbf{ No Scaling}$', plot_comments='_unscaledlowAoL', fs=(7,5), plot_dir=plot_dir)
    # plot_Glegacy(data1b=data1b, save_figs=save_figs, lAoLscale=lAoLscale, title=title+'$\\textbf{A/L }\\mathbf{<}\\textbf{ 1um scaled x'+str(lAoLscale)+'}$', plot_comments=plot_comments, fs=(7,5), plot_dir=plot_dir)


if lit_compare:

    ### load fit results
    if calc=='Mean':
        print('\nCalculating fit parameters as the mean of the simulation values.\n')
        sim_results = [np.mean(sim_data, axis=0), np.std(sim_data, axis=0)]
    elif calc=='Median':
        print('\nCalculating fit parameters as the median of the simulation values.\n')
        sim_results = [np.median(sim_data, axis=0), np.std(sim_data, axis=0)]
    else:
        print('Unknown parameter calculation method. Choose "mean" or "median".')


    Gmeas_U, Gmeas_W, Gmeas_I, alpham_U, alpham_W, alpham_I = sim_results[0]; sigGU, sigGW, sigGI, sigalpha, sigalphaW, sigalphaI = sim_results[1]
    kappam_U = GtoKappa(Gmeas_U, alpham_U, L); sigkappam_U = GtoKappa(sigGU, alpham_U, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
    kappam_W = GtoKappa(Gmeas_W, alpham_W, L); sigkappam_W = GtoKappa(sigGW, alpham_W, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
    kappam_I = GtoKappa(Gmeas_I, alpham_I, L); sigkappam_I = GtoKappa(sigGI, alpham_I, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants

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

    f = 1.   # total specular scattering
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


if design_implications:   # TES and NIS design implications of this work

    ### plot G_TES and TFN as a function of substrate width
    ### assumes two legs with full microstrip and two are 
    # with open(sim_file, 'rb') as infile:   # load simulation pkl
    #     sim_dict = pkl.load(infile)
    simresults_mean = np.array([np.mean(sim_data, axis=0), np.std(sim_data, axis=0)])
    simresults_med = np.array([np.median(sim_data, axis=0), np.std(sim_data, axis=0)])
    sim_results = simresults_mean if calc=='Mean' else simresults_med

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


if analyze_vlengthdata:

    title = 'Predictions for Bolotest Data, '   
    plot_modelvdata(sim_data, data, title=title+' $\\sigma_G$ = 2.4\\% G', plot_bolotest=True, layer_ds=layer_d0, pred_wfit=False)


if manual_params:

    params_manual_params = np.array([0.7, 0.8, 1.3, 1, 0.5, 1.2])
    sigma_params = np.array([0, 0, 0, 0, 0, 0])
    fit = np.array([params_manual_params, sigma_params])
    title = 'Hand-chosen fit parameters '   
    data1b = np.array([ydata[0], sigma[0]]) if plot_bolo1b else []  # plot bolo1b data?

    results = qualityplots(data, params_manual_params, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments, title=title, vmax=1E3, calc='Mean', qplim=qplim)

    predict_Glegacy(fit, data1b=data1b, save_figs=save_figs, title=title, plot_comments=fn_comments, fs=(7,7))
    plot_modelvdata(fit, data, title=title+' L=220 um Bolos', plot_bolotest=True)
    plot_modelvdata(fit, data, title=title+' Bolos 1a-f and bolotest', vlength_data=vlength_data, plot_bolotest=False)
    plot_modelvdata(fit, data, title=title+' Bolos 1a-f and bolotest, $\\beta=0.8$', vlength_data=vlength_data, plot_bolotest=False, Lscale=0.8)


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