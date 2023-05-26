"""
Script for measuring individual layer contributions to the total thermal conductivity of a TES. 
Data from 'bolotest' measurements run in 2018. 
G_total and error is from TES power law fit measured at Tc and scaled to 170 mK, assuming dP/dT = G = n*k*Tc^(n-1).
Layers: U = SiN + SiO substrate, W = Nb wiring layers (<=2 layers per leg), I = SiN insulating layers (<=2 layers per leg)

aharkehosemann@gmail.com
2022/01


UPDATES

2022.06.15 : We recently discovered the wiring layers have a smaller width (W1=5um, W2=3um) than the SiN layers, and that there's a 120nm SiOx layer on the membrane under the 300nm SiNx layer. 
I now scale W2 terms by 3/5 and W stack by thickness-weighted average width (assumes G scales linearly with layer width). 
Shannon estimates 11-14 nm etching on the SiOx, so true thickness is ~109 nm. 

2022.06.28 : Should we model the data as five layers (treat substrate as 420 nm thick layer of one material) or six layers (one 120 nm SiOx and one 300nm layer SiNx)?
DOF of six-layer fit is 0...
What I've been referring to as "function to minimize" is actually weighted least squares. 
Not sure if looking at significance is useful because we aren't hypothesis testing and have very small DOF.

2022/08/03 : Did full error analysis to get sigma_G on G(170), i.e. I wrote sigma_GscaledT() and added those errors to the spreadsheet, where I then took the weighted average and error if there are redundant bolos.

2022/10/07 : Previously the "unconstrained model" actually allowed alpha to vary from 0 to 2. If allowed to vary between [0,inf), the alpha_W error is much larger. 

2022/11/15 : For bolotest-like G predictions and NEP, I changed the W1 and W2 width scaling for leg widths<7 um to be lw - 2 um and lw - 4 um respectively. 
Previously G(microstrip) was scaled by leg width/7 for lw<7um and was the same value for lw>7, which does not take into account that I widens with the leg.

2023/04/19: sigma_current in IV has been overestimated by a couple orders of magnitude (was not subtracting line fit in normal branch). Reran sim with this correction on 5/5.  

2023/05/10: n term in sigma_G calculation was missing a component and is now expected to dominate sigma_G.    

TODO: redo six-layer fit to compare chi-squared and WLS values?; is NA correct in calculation of C_V?
"""
from bolotest_routines import *
from scipy.optimize import fsolve
import csv 


### User Switches
# choose analysis
run_sim = True   # run MC simulation for fitting model
quality_plots = True   # results on G_x vs alpha_x parameter space for each layer
random_initguess = False   # try simulation with randomized initial guesses
average_qp = False   # show average value for 2D quality plot over MC sim to check for scattering
lit_compare = False   # compare measured conductivities with literature values
compare_legacy = False   # compare with NIST sub-mm bolo legacy data
design_implications = False
load_and_plot = False   # scrap; currently replotting separate quality plots into a 1x3 subplot figure
scrap = False

# options
save_figs = True   
save_sim = True   # save full simulation
save_csv = True   # save csv file of resulting parameters
show_plots = True   # show simulated y-data plots during MC simulation
calc_Gwire = False   # calculate G of the wiring stack if it wasn't saved during the simulation
latex_fonts = True


n_its = int(1E4)   # number of iterations for MC simulation
num_guesses = 100   # number of randomized initial guesses to try
analysis_dir = '/Users/angi/NIS/Bolotest_Analysis/'
# fn_comments = '_alpha0inf_1E5iterations'   # incorrect current error calc
# fn_comments = '_alpha0inf_1E4iterations_updatedIerrors'   # correct current error but incorrect n term in sigma_GTES
# fn_comments = '_alpha0inf_1E4iterations_updatednterminGerror'   # corrected n term
# fn_comments = '_alpha0inf_1E4iterations_noPerror'   # fitter does not know P measurement error
# fn_comments = '_alpha0inf_1E4iteratinos_firsterror'   # first go at getting errors, incorrect sigma_I (mean not subtracted from and n-term in sigma_G)
fn_comments = '_alpha0inf_1E4iteratinos_firsterror_recalculatedsigmaGTES'   # first go at getting errors, errors pulled from _errorcompare_master, incorrect sigma_I (mean not subtracted from and n-term in sigma_G)
# fn_comments = '_alpha0inf_1E4iterations_wrongsigmaI'   # incorrect sigma_I
# fn_comments = '_alpha0inf_1E4iterations_wrongsigmaG_3ndrun'   # correct sigma_i, incorrect sigma_G n-term
# fn_comments = '_alpha0inf_1E5iterations_wrongsigmaG'   # correct sigma_i, incorrect sigma_G n-term
# fn_comments = '_alpha0ainf_1E4iterations_corrsigmaG_2ndrun'   # corrected n term, pretty stable run to run
# fn_comments = '_alpha0inf_1E4iterations_nobling'   # corrected n term and exclude extra bling bolos
plot_comments = ''
alim = [0,np.inf]   # limits for fitting alpha
# alim = [0,1]   # limits for fitting alpha
L = 220   # TES leg length, um
A_U = 7*420E-3; A_W = 5*400E-3; A_I = 7*400E-3  # Area of film on one leg,um^2

# initial guess for fitter
# p0_a0inf = np.array([0.74, 0.59, 1.29, 0.47, 1.94, 1.26]); sigmap0_a0inf = np.array([0.09, 0.19, 0.06, 0.58, 2.45, 0.11])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]; fit vals from alpha=[0,2] model
p0_a0inf_median = np.array([0.73, 0.61, 1.28, 0.20, 1.2, 1.25]);# sigmap0_a0inf_median = np.array([0.09, 0.19, 0.06, 0.58, 2.45, 0.11])   # U, W, I [pW/K], alpha_U, alpha_W, alpha_I [unitless]; fit vals from alpha=[0,2] model
# p0_a02 = np.array([0.77, 0.52, 1.28, 0.63, 1.23, 1.26]); sigmap0_a02 = np.array([0.12, 0.20, 0.05, 0.73, 0.80, 1.26])   # fit vals from alpha=[0,2] model
p0_a01 = np.array([0.8, 0.42, 1.33, 1., 1., 1.]); sigmap0_a01 = np.array([0.03, 0.06, 0.03, 0.02, 0.03, 0.00])   #  fit vals from alpha=[0,2] model, 1E5 iterations
p0 = p0_a0inf_median;# sigma_p0 = sigmap0_a0inf_median

# choose data to use
# ydata_Tc = np.array([11.7145073, 4.921841228, 8.077815536, 10.03001622, 16.63099617, 5.386790491, 15.2863792, 3.585251305])   # pW/K at Tc, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13*; bolo1b is weighted average (not sure we trust row 10)
# sigma_Tc = np.array([0.100947739, 0.063601732, 0.078665632, 0.130040288, 0.142600818, 0.059261252, 0.123206779, 0.052084114])   # pW/K at Tc; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13*
ydata_170mK_noPerr = [13.87604963, 5.11402462, 8.285371853, 10.28612903, 16.69684211, 5.501194508, 15.48321544, 3.576823158]   # pW/K at 170 mK fit without P error, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13* (only using row 5 value of bolo1b)
sigma_170mK_noPerr = [0.814944269, 0.218018499, 0.29514379, 0.620991797, 1.09788405, 0.175358636, 1.054916235, 0.090860025]
# ydata_170mK_witherr = [13.51632171, 4.889791292, 7.929668225, 9.925580294, 16.27276237, 5.27649525, 14.95079826, 3.577979915]   # pW/K at 170 mK with full G temp scaling error analysis, pulled from _witherrors spreadsheet, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13* (only using row 5 value of bolo1b)
# sigma_170mK_witherr = [0.396542, 0.08774166, 0.148831461, 0.22016008, 0.411908086, 0.079748858, 0.340424395, 0.074313687]
ydata_170mK_witherr = [13.51632171, 4.889791292, 7.929668225, 9.925580294, 16.27276237, 5.27649525, 14.95079826, 3.577979915]   # pW/K at 170 mK with full G temp scaling error analysis, errors calculated in _errorcompare_master, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13* (only using row 5 value of bolo1b)
sigma_170mK_witherr = [0.396542, 0.066131706, 0.127251408, 0.22016008, 0.34608384, 0.069631337, 0.302880201, 0.053104708]
ydata_170mK_wrongsigmaI = [13.51321747, 4.93174278, 8.000974716, 9.917328335, 16.32689035, 5.348468327, 15.00602246, 3.577102069]   # pW/K at 170 mK with full G temp scaling error analysis, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13* (only using row 5 value of bolo1b)
sigma_170mK_wrongsigmaI = [34.762719, 12.42239969, 15.1839926, 26.51034923, 25.99331922, 10.8398234, 22.41338756, 9.70233484]
ydata_170mK_wrongsigmaG = [13.24451089, 5.000777852, 7.960183617, 9.833994168, 15.72463397, 5.254644954, 14.31984759, 3.577102069]   # pW/K at 170 mK with corrected current noise measurement, weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13* (only using row 5 value of bolo1b)
sigma_170mK_wrongsigmaG = [0.381955399, 0.063312254, 0.08912874, 0.181640644, 0.383600005, 0.058708221, 0.44963107, 0.043775235]
ydata_170mK_corrsigmaG = [13.24451089, 4.999984304, 7.960081461, 9.833994168, 15.72503124, 5.257487318, 14.31854675, 3.581627355]   # pW/K at 170 mK with fixed n term in GTES error, most were weighted averages*; bolo 1b*, 24*, 23*, 22, 21*, 20*, 7*, 13* (only using row 5 value of bolo1b)
sigma_170mK_corrsigmaG = [0.49306722, 0.078336521, 0.112242869, 0.231738644, 0.50525562, 0.073826253, 0.58107854, 0.053441225]
ydata_170mK_nobling = [13.24451089, 4.668924466, 7.830962629, 9.833994168, 15.57684727, 5.198992752, 14.31854675, 3.460230096]   # pW/K at 170 mK with fixed n term in GTES error, weighted average only on 7 (second vals have extra bling); bolo 1b, 24, 23, 22, 21, 20, 7*, 13 
sigma_170mK_nobling = [0.49306722, 0.142644772, 0.17760485, 0.231738644, 0.808761146, 0.080176748, 0.58107854, 0.076007386]
ydata = np.array(ydata_170mK_witherr); sigma = np.array(sigma_170mK_witherr)

bolos = np.array(['bolo 1b', 'bolo 24', 'bolo 23', 'bolo 22', 'bolo 21', 'bolo 20', 'bolo 7', 'bolo 13'])
bounds = [(0, np.inf), (0, np.inf), (0, np.inf), (alim[0], alim[1]), (alim[0], alim[1]), (alim[0], alim[1])]   # bounds for 6 fit parameters: G_U, G_W, G_I, alpha_U, alpha_W, alpha_I
# fits_J = np.array([.89, 0.30, 1.3277, 0.58, 1.42, 1.200]); error_J = np.array([0.14, 0.18, 0.02, 0.57, 0.53, 0.06])   # compare with Joel's results
# results_J = [fits_J, error_J]   # Joel's fit parameters and error
# boundsred = [(0, np.inf), (alim[0], alim[1])]   # G, alpha
plot_dir = analysis_dir + 'plots/layer_extraction_analysis/'
sim_file = analysis_dir + 'Analysis_Files/sim' + fn_comments + '.pkl'
csv_file = analysis_dir + 'Analysis_Files/sim' + fn_comments + '.csv'
data = [ydata, sigma] 

# Gfile = '/Users/angi/NIS/Bolotest_Analysis/Analysis_Files/'
# dfile = dfiles[bb]
# with open(dfile, "rb") as f:
#     data_temp = pkl.load(f, encoding='latin1')
# data[bays[bb]] = data_temp[bays[bb]]

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
    sim_results = runsim_WLS(n_its, p0, data, bounds, plot_dir, show_yplots=show_plots, save_figs=save_figs, save_sim=save_sim, sim_file=sim_file, fn_comments=fn_comments)  


if quality_plots:   # plot G_x vs alpha_x parameter space with various fit results

    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
    sim_data = sim_dict['sim']
    params_mean = [sim_dict['fit']['fit_params'], sim_dict['fit']['fit_std']]
    params_median = [np.median(sim_data, axis=0), np.std(sim_data, axis=0)]   # take median value to avoid outliers
    
    if calc_Gwire:   # just in case Gwires wasn't saved to simualtion file
        Gwires = np.array([Gfrommodel(row, 0.420, 7, 220, layer='wiring', fab='bolotest')[0,0]/4 for row in sim_data])   # this could be better but works for now
    else:
        Gwires = sim_dict['Gwires']
    Gwire_mean = np.mean(Gwires); Gwire_median = np.median(Gwires); sigma_Gwire = np.std(Gwires)

    if np.isinf(alim[1]):   # quality plot title
        qp_title = '$\\boldsymbol{\\mathbf{\\alpha \\in [0,\infty)}}$\\textbf{ (Mean)}'   # title for 1x3 quality plots
    else:
        qp_title = '$\\boldsymbol{\\mathbf{\\alpha \in [0,'+str(alim[1])+']}}$'   # title for 1x3 quality plots

    ### plot fit in 2D parameter space
    fn_comments2 = fn_comments+'_mean'
    qualityplots(data, params_mean, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments2, title=qp_title)

    ### calculate thermal conductivities
    print ('\n\nResults taking mean values of fit parameters:')
    GmeasU_mean, GmeasW_mean, GmeasI_mean, alphaU_mean, alphaW_mean, alphaI_mean = params_mean[0]; sigGU, sigGW, sigGI, sigalphaU, sigalphaW, sigalphaI = params_mean[1]
    print('G_U(420 nm) = ', round(GmeasU_mean, 2), ' +/- ', round(sigGU, 2), 'pW/K')
    print('G_W(400 nm) = ', round(GmeasW_mean, 2), ' +/- ', round(sigGW, 2), 'pW/K')
    print('G_I(400 nm) = ', round(GmeasI_mean, 2), ' +/- ', round(sigGI, 2), 'pW/K')
    print('alpha_U = ', round(alphaU_mean, 2), ' +/- ', round(sigalphaU, 2))
    print('alpha_W = ', round(alphaW_mean, 2), ' +/- ', round(sigalphaW, 2))
    print('alpha_I = ', round(alphaI_mean, 2), ' +/- ', round(sigalphaI, 2))
    print('')
    kappaU_mean = GtoKappa(GmeasU_mean, A_U, L); sigkappaU_mean = GtoKappa(sigGU, A_U, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
    kappaW_mean = GtoKappa(GmeasW_mean, A_W, L); sigkappaW_mean = GtoKappa(sigGW, A_W, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
    kappaI_mean = GtoKappa(GmeasI_mean, A_I, L); sigkappaI_mean = GtoKappa(sigGI, A_I, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
    print('Kappa_U: ', round(kappaU_mean, 2), ' +/- ', round(sigkappaU_mean, 2), ' pW/K/um')
    print('Kappa_W: ', round(kappaW_mean, 2), ' +/- ', round(sigkappaW_mean, 2), ' pW/K/um')
    print('Kappa_I: ', round(kappaI_mean, 2), ' +/- ', round(sigkappaI_mean, 2), ' pW/K/um')
    print('G_wire = ', round(Gwire_mean, 2), ' +/- ', round(sigma_Gwire, 2), 'pW/K')

    WLS_fit = WLS_val(params_mean[0], data)
    print('WLS value for the fit: ', round(WLS_fit, 3)) 
    chisq_fit = calc_chisq(ydata, Gbolos(params_mean[0]))
    print('Chi-squared value for the fit: ', round(chisq_fit, 3))
    vals_mean = np.array([params_mean[0][0], params_mean[0][1], params_mean[0][2], params_mean[0][3], params_mean[0][4], params_mean[0][5], kappaU_mean, kappaW_mean, kappaI_mean, Gwire_mean, WLS_fit, chisq_fit])
    vals_err = np.array([params_mean[1][0], params_mean[1][1], params_mean[1][2], params_mean[1][3], params_mean[1][4], params_mean[1][5], sigkappaU_mean, sigkappaW_mean, sigkappaI_mean, sigma_Gwire, '', ''])   # should be the same for mean and median

    print ('\n\nResults taking median values of fit parameters:')
    if np.isinf(alim[1]):   # quality plot title
        qp_title = '$\\boldsymbol{\\mathbf{\\alpha \\in [0,\infty)}}\\textbf{ (Median)}$'   # title for 1x3 quality plots
    else:
        qp_title = '$\\boldsymbol{\\mathbf{\\alpha \in [0,'+str(alim[1])+']}}$'   # title for 1x3 quality plots


    ### plot fit in 2D parameter space using median values
    fn_comments3 = fn_comments+'_median'
    qualityplots(data, params_median, plot_dir=plot_dir, save_figs=save_figs, fn_comments=fn_comments3, title=qp_title)

    ### calculate thermal conductivities
    GmeasU_med, GmeasW_med, GmeasI_med, alphaU_med, alphaW_med, alphaI_med = params_median[0]; sigGU, sigGW, sigGI, sigalphaU, sigalphaW, sigalphaI = params_median[1]
    print('G_U(420 nm) = ', round(GmeasU_med, 2), ' +/- ', round(sigGU, 2), 'pW/K')
    print('G_W(400 nm) = ', round(GmeasW_med, 2), ' +/- ', round(sigGW, 2), 'pW/K')
    print('G_I(400 nm) = ', round(GmeasI_med, 2), ' +/- ', round(sigGI, 2), 'pW/K')
    print('alpha_U = ', round(alphaU_med, 2), ' +/- ', round(sigalphaU, 2))
    print('alpha_W = ', round(alphaW_med, 2), ' +/- ', round(sigalphaW, 2))
    print('alpha_I = ', round(alphaI_med, 2), ' +/- ', round(sigalphaI, 2))
    print('')    
    kappaU_med = GtoKappa(GmeasU_med, A_U, L); sigkappaU_med = GtoKappa(sigGU, A_U, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
    kappaW_med = GtoKappa(GmeasW_med, A_W, L); sigkappaW_med = GtoKappa(sigGW, A_W, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
    kappaI_med = GtoKappa(GmeasI_med, A_I, L); sigkappaI_med = GtoKappa(sigGI, A_I, L)   # pW / K / um; error analysis is correct because kappa(G) just depends on constants
    print('Kappa_U: ', round(kappaU_med, 2), ' +/- ', round(sigkappaU_med, 2), ' pW/K/um')
    print('Kappa_W: ', round(kappaW_med, 2), ' +/- ', round(sigkappaW_med, 2), ' pW/K/um')
    print('Kappa_I: ', round(kappaI_med, 2), ' +/- ', round(sigkappaI_med, 2), ' pW/K/um')
    print('G_wire = ', round(Gwire_median, 2), ' +/- ', round(sigma_Gwire, 2), 'pW/K')

    WLS_fit = WLS_val(params_median[0], data)
    print('WLS value for the fit: ', round(WLS_fit, 3))   
    chisq_fit = calc_chisq(ydata, Gbolos(params_median[0]))
    print('Chi-squared value for the fit: ', round(chisq_fit, 3))
    vals_med = np.array([params_median[0][0], params_median[0][1], params_median[0][2], params_median[0][3], params_median[0][4], params_median[0][5], kappaU_med, kappaW_med, kappaI_med, Gwire_median, WLS_fit, chisq_fit])

    if save_csv:
        # write CSV     
        csv_params = np.array(['GU (pW/K)', 'GW (pW/K)', 'GI (pW/K)', 'alphaU', 'alphaW', 'alphaI', 'kappaU (pW/K/um)', 'kappaW (pW/K/um)', 'kappaI (pW/K/um)', 'Gwire (pW/K)', 'WLS val', 'Chi-sq val'])
        fields = np.array(['Parameter', 'Mean', 'Median', 'Error'])  
        rows = [[csv_params[rr], vals_mean[rr], vals_med[rr], vals_err[rr]] for rr in np.arange(len(csv_params))]
        with open(csv_file, 'w') as csvfile:  
            csvwriter = csv.writer(csvfile)  # csv writer object  
            csvwriter.writerow(fields)  
            csvwriter.writerows(rows)


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

    ### load fit results
    with open(sim_file, 'rb') as infile:   # load simulation pkl
        sim_dict = pkl.load(infile)
    if 'inf' in sim_file:
        sim_results = [np.median(sim_dict['sim'], axis=0), np.std(sim_dict['sim'], axis=0)]   # take median value instead of mean for alpha=[0,inf] model
    else:
        sim_results = [sim_dict['fit']['fit_params'], sim_dict['fit']['fit_std']]

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

    L = 220   # bolotest leg length, um
    boloGs = ydata_170mK; sigma_boloGs = sigma_170mK   # choose bolotest G's to compare
    wstack_width = (5*0.100+3*0.285)/(0.100+0.285)   # um, effective width of W1 W2 stack on bolo 20
    A_bolo = np.array([(7*4*.420+5*4*.160+3*4*.340+7*4*.350+7*4*.400), (7*1*.420+7*3*.340+5*.160+3*.340+7*.350+7*.400), (7*2*.420+7*2*.340+5*2*.160+3*2*.340+7*2*.350+7*2*.400), (7*3*.420+7*1*.340+5*3*.160+3*3*.340+7*3*.350+7*3*.400), (7*1*.420+7*3*.400+5*1*.160+3*1*.285+7*3*.370+7*1*.350), (7*4*.420+5*1*.160+wstack_width*3*.385+3*1*.285+7*1*.340), (7*3*.420+7*1*.400+5*3*.160+3*1*3.340+7*3*.350+7*1*.670+7*3*.400), (7*1*.420+7*3*.400+5*1*.160+3*1*.285+7*1*.350) ])   # bolotest areas
    # ASiN = np.array([(7*4*.300), (7*1*.300+7*3*.220), (7*2*.300+7*2*.220), (7*3*.300+7*1*.220), (7*1*.300+7*3*.280), (7*4*.300), (7*3*.300+7*1*.280), (7*1*.300+7*3*.280)])   # area of 4 nitride beams
    AoL_bolo = A_bolo/L   # A/L for bolotest devices

    dW1 = .190; dI1 = .350; dW2 = .400; dI2 = .400   # general film thicknesses, um
    # dmicro = dW1 + dI1 + dW2 + dI2   
    legacy_Gs = np.array([1296.659705, 276.1, 229.3, 88.3, 44, 76.5, 22.6, 644, 676, 550, 125, 103, 583, 603, 498, 328, 84, 77, 19, 12.2, 10.5, 11.7, 13.1, 9.98, 16.4, 8.766, 9.18, 8.29, 9.57, 7.14, 81.73229733, 103.2593154, 106.535245, 96.57474779, 90.04141806, 108.616653, 116.2369491, 136.2558345, 128.6066776, 180.7454359, 172.273248, 172.4456603, 192.5852409, 12.8, 623, 600, 620, 547, 636, 600.3, 645, 568, 538.7, 491.3, 623, 541.2, 661.4, 563.3, 377.3, 597.4, 395.3, 415.3, 575, 544.8, 237.8, 331.3, 193.25, 331.8, 335.613, 512.562, 513.889, 316.88, 319.756, 484.476, 478.2, 118.818, 117.644, 210.535, 136.383, 130.912, 229.002, 236.02, 101.9, 129.387, 230.783, 230.917, 130.829, 127.191, 232.006, 231.056])  
    legacy_ll = np.array([250, 61, 61, 219.8, 500, 500, 1000, 50, 50, 50, 100, 300, 50, 50, 50, 100, 100, 300, 500, 1000, 1000, 1000, 1000, 1250, 500, 1000, 1000, 1000, 1000, 1250, 640, 510, 510, 510, 510, 510, 730, 610, 490, 370, 370, 370, 300, 500, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
    legacy_lw = np.array([25, 14.4, 12.1, 10, 10, 15, 10, 41.5, 41.5, 34.5, 13, 16.5, 41.5, 41.5, 34.5, 29, 13, 16.5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 7, 7, 7, 7, 7, 7, 10, 10, 8, 8, 8, 8, 8, 6, 41.5, 41.5, 41.5, 41.5, 41.5, 41.5, 41.5, 41.5, 34.5, 34.5, 41.5, 41.5, 34.5, 34.5, 23.6, 37.5, 23.6, 23.6, 37.5, 37.5, 13.5, 21.6, 13.5, 21.6, 23.6, 37.5, 37.5, 23.6, 23.6, 37.5, 37.5, 11.3, 11.3, 18.5, 11.3, 11.3, 18.5, 18.5, 11.3, 11.3, 18.5, 18.5, 11.3, 11.3, 18.5, 18.5])
    legacy_dsub = 0.450 + np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    legacy_Tcs = np.array([557, 178.9, 178.5, 173.4, 170.5, 172.9, 164.7, 163, 162, 163, 164, 164, 168, 168, 167, 167, 165, 166, 156, 158, 146, 146, 149, 144, 155, 158, 146, 141, 147, 141, 485.4587986, 481.037173, 484.9293596, 478.3771521, 475.3010335, 483.4209782, 484.0258522, 477.436482, 483.5417917, 485.8804622, 479.8911157, 487.785816, 481.0323883, 262, 193, 188, 188.8, 188.2, 190.2, 188.1, 186.5, 184.5, 187.5, 185.5, 185.8, 185.6, 185.7, 183.3, 167.3, 167, 172.9, 172.8, 166.61, 162.33, 172.87, 161.65, 163.06, 166.44, 177.920926, 178.955154, 178.839062, 177.514658, 177.126927, 178.196297, 177.53632, 169.704602, 169.641018, 173.026393, 177.895192, 177.966456, 178.934122, 180.143125, 177.16833, 178.328865, 179.420334, 179.696264, 172.724501, 172.479515, 177.385267, 177.492689])*1E-3
    legacy_ns = np.array([3.5, 3.4, 3.4, 3, 2.8, 3, 2.7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2.707252717, 2.742876666, 2.741499631, 2.783995279, 2.75259088, 2.796872814, 2.747211811, 2.782265754, 2.804876038, 2.879595447, 2.871133545, 2.889243695, 2.870571891, 2.6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    legacy_Gs170 = scale_G(.170, legacy_Gs, legacy_Tcs, legacy_ns)   # scale G's to Tc of 170 mK
    # legacy_A = 4*(legacy_dsub + dmicro)*legacy_lw   # um^2, area for four legs, thickness is substrate + wiring stack
    legacy_w1w, legacy_w2w = wlw(legacy_lw)
    legacy_A = 4*(legacy_dsub + dI1 + dI2)*legacy_lw + 4*dW1*legacy_w1w + 4*dW2*legacy_w2w   # um^2, area for four legs, thickness is substrate + wiring stack
    legacy_AoLs = legacy_A/legacy_ll  # um
     
    lTcinds = np.where(legacy_Tcs<0.200)[0]   # Tc<200 mK bolometers from Shannon's data
    hTcinds = np.where(legacy_Tcs>=0.200)[0]   # Tc<200 mK bolometers from Shannon's data
    legacy_geom = np.array([legacy_dsub[lTcinds], legacy_lw[lTcinds], legacy_ll[lTcinds]])
    legacy_data = np.array([legacy_AoLs[lTcinds], legacy_Gs170[lTcinds]])

    title='Predictions from Model, Restricted $\\alpha$'
    plot_comments = '_a01_mean'
    fit = ([p0_a01, sigmap0_a01])
    predict_G(fit, legacy_data, legacy_geom, bolo1b=False, save_figs=save_figs, title=title, plot_comments=plot_comments, fs=(7,7))

    title='Predictions from Model, Unrestricted $\\alpha$'
    plot_comments = '_a0inf_median'
    fit = ([p0_a0inf_median, sigmap0_a0inf_median])
    predict_G(fit, legacy_data, legacy_geom, bolo1b=False, save_figs=save_figs, title=title, plot_comments=plot_comments)
    (G_layer(fit, dI1, layer='I') + G_layer(fit, dI2, layer='I')) *lw/7 *220/ll
    # title='Predictions from Model, Unrestricted $\\alpha$ (Mean)'
    # plot_comments = '_a0inf_mean'
    # fit = ([p0_a0inf, sigmap0_a0inf])
    # predict_G(fit, legacy_data, legacy_geom, bolo1b=False, save_figs=save_figs, title=title, plot_comments=plot_comments)

    title="Predictions from Layer $\kappa$'s"
    plot_comments = '_kappa'
    predict_G(fit, legacy_data, legacy_geom, bolo1b=False, save_figs=save_figs, estimator='kappa', title=title, plot_comments=plot_comments)


if design_implications:   # making plots to illustrate TES and NIS design implications of this work


    ### plot G_TES and TFN as a function of substrate width
    # fit_params = p0_a0inf_median; sig_params = sigmap0_a0inf_median
    fit = np.array([p0_a0inf_median, sigmap0_a0inf_median])
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

    G_full, Gerr_full = Gfrommodel(fit, dsub, lwidths, llength, layer='total', fab='bolotest')
    G_U, Gerr_U = Gfrommodel(fit, dsub, lwidths, llength, layer='U', fab='bolotest')
    G_W1, Gerr_W1 = Gfrommodel(fit, dsub, lwidths, llength, layer='W1', fab='bolotest')
    G_Nb200 = G_U+G_W1; Gerr_Nb200 = Gerr_U+Gerr_W1
    G_bare, Gerr_bare = Gfrommodel(fit, .340, lwidths, llength, layer='U', fab='bolotest')   # bare substrate is thinner from etching steps

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
    xover_dsub = fsolve(xover_finder, ll_test, args=(fit, lw_test, ll_test))[0]



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

    print('Results from Monte Carlo Sim - WLS Min')
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

    WLS_fit = WLS_val(sim_results[0], data)
    print('WLS value for the fit: ', round(WLS_fit, 3))   # the function we've been minimizing is not actually the chi-squared value...

    chisq_fit = calc_chisq(ydata, Gbolos(sim_results[0]))
    print('Chi-squared value for the fit: ', round(chisq_fit, 3))
    
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
    sim_result = minimize(WLS_val, p0, args=[ydata, sigma], bounds=bounds)
    curve_fit(self.powerlaw_fit_func, temperatures, powerAtRns[index], p0=init_guess, sigma=sigma[index], absolute_sigma=True) 

plt.show()