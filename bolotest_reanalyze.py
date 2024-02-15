""" 
Reanalyze bolotest data from 2018 (?)
AHHH 2020/12

to do: fix SC branch of IVs; second y axis on Psat plot
automated error analysis: baseline removal tes-by-tes basis scipy.signal.sabgal_filter
"""
import numpy as np 
import matplotlib.pyplot as plt
import pickle as pkl
from tesanalyze_ahhh import *
import pdb
import csv 
from collections import OrderedDict

### user params
# which analysis?
analyze_ivs = False
estimate_Gcov = False   # estimating n, k, G covariance to inform error analysis
plot_noisyivs = False
plot_mbolos = False
save_pad18 = False   # write quick CSV for Joel
scaleG = False   # scale G(Tc) to G(170 mK)
plot_bolo23 = True

# analysis options
save_data = False   # overwrite csv and pkl data files
remove_flagged = True   # flag and remove IV data points between SC branch and transition
constT = True   # save constant T_TES assumption results
fitGexplicit = True   # fit G explicitly

# plot options
show_figs = True   # show plots after running
save_figs = False  # overwrite figures
iv_plots = False   # make plots of raw IV to real units - turn off if analyzing all TESs
branch_plots = False   # make plots showing SC and N branch fits - turn off if analyzing all TESs
interp_plots = False   # make with all Tb IVs and interpolated points
plot_byind = False   # add IV index to plots for excluding noisy IVs
latex_fonts = False
psat_calc = False   # show where on the IV Psat is calculated; I've never seen this not work

### data and analysis files
bolotest_dir = '/Users/angi/NIS/Bolotest_Analysis/'
# fn_comments = '_fitGexplicit_constTTES_rerun'   # fit G explicitly, assume constant T_TES regardless of %Rn
# fn_comments = '_reanalyzed_SCbranch'   # fit G explicitly, assume constant T_TES regardless of %Rn
# fn_comments = '_reanalyzed_testscriptchanges'   # fit G explicitly, assume constant T_TES regardless of %Rn
fn_comments = '_reanalyzed_cleaningupcode'   # fit G explicitly, assume constant T_TES regardless of %Rn
dfiles = ['/Users/angi/NIS/Bolotest_Analysis/Data/MM2017L_20171117_data/AY_1thru15_IVvsTb.pkl', '/Users/angi/NIS/Bolotest_Analysis/Data/MM2017L_20171117_data/AX_16thru30_IVvsTb.pkl']
csv_file = bolotest_dir + 'Analysis_Files/bolotest_reanalyzedAHHH_202309' + fn_comments + '.csv'   # where to save analysis results
pkl_file = bolotest_dir + 'Analysis_Files/bolotest_reanalyzedAHHH_202309' + fn_comments + '.pkl'   # where to save analysis results

### analysis options 
perRn = np.array([25, 30, 40, 50, 60, 70, 80])   # % Rn
perRn_toquote = 80   # % Rn fit values to print to screen
Tb_inds = np.array([0, 3, 8, 11, 16])   # fit G etc at these Tbaths
Tbq_ind = 16   # quote G etc at this Tbath (168 mK); there isn't always a Tbath = 165 mK measurement
v_nfit = .3   # v_bias above which TES is normal (approximate)
tran_perRn_start = 0.2   # Fraction of Rn dubbed beginning of SC transition
init_guess = np.array([1.E-10, 2.5, .170]) if not fitGexplicit else np.array([10E-12, 2.5, .170])  # kappa, n, Tc [mK]; powerlaw fitter

### readout circuit parameters
M_r = 8.5   # (SQUID-input coupling: current = arbs / (M_r * R_fb))
Rfb = 2070.   # Ohms (SQUID feeback resistor), Kelsey: accurate to ~5%
Rsh = 370.e-6   # microOhms (bias circuit shunt), Kelsey: accurate to 10-15%
Rb = 1020.   # Ohms (TES bias resistor), Kelsey: accurate to ~5%

pad_bolo_map = {'1':'1f', '2':'1e', '3':'1d', '4':'1c', '5':'1b', '6':'1a', '7':'1f', '8':'1e', '9':'1d', '10':'1c', '11':'1b', '12':'1a', 
                '13':'24', '14':'23', '15':'22', '16':'13', '17':'21', '18':'20', '19':'24', '20':'23', '21':'22', '22':'13', '23':'21', 
                '24':'20','25':'7','26':'26','27':'27','28':'7','29':'26','30':'27'}

if latex_fonts:   # fonts for paper plots
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=18)
    plt.rc('font', weight='normal')
    plt.rcParams['text.latex.preamble']="\\usepackage{amsmath}"
    plt.rcParams['xtick.major.size'] = 5; plt.rcParams['xtick.minor.visible'] = False    
    plt.rcParams['ytick.major.size'] = 5; plt.rcParams['ytick.minor.visible'] = False

if analyze_ivs:


    data, tesids = load_data(dfiles)
    ivs = convert_rawdata(data, tesids, constT=constT, v_nfit=v_nfit, save_figs=save_figs,  branch_plots=branch_plots, iv_plots=iv_plots, tran_perRn_start=tran_perRn_start, perRn=perRn, 
                        fn_comments=fn_comments,rm_fl=remove_flagged, bolotest_dir=bolotest_dir, interp_plots=interp_plots)
    ivs = fit_powerlaws(ivs, save_figs=save_figs, fitGexplicit=fitGexplicit, perRn_toquote=perRn_toquote, Tb_inds=Tb_inds, fn_comments=fn_comments, show_psatcalc=psat_calc, Tbq_ind=Tbq_ind, constT=constT, 
                      perRn=perRn, init_guess=init_guess, bolotest_dir=bolotest_dir)
    if save_data: ivs = savedata(ivs, pkl_file, csv_file)

if plot_noisyivs:

    noisy_bolos = ['BayAY_Row00', 'BayAY_Row09', 'BayAY_Row15', 'BayAX_Row05', 'BayAX_Row11', 'BayAX_Row14', 'BayAX_Row15']

    ivs = {} 
    # tesids = []
    data = {}
    for bb in np.arange(len(bays)):
        ### load data
        dfile = dfiles[bb]
        with open(dfile, "rb") as f:
            data_temp = pkl.load(f, encoding='latin1')
        data[bays[bb]] = data_temp[bays[bb]]

    for ts, tesid in enumerate(noisy_bolos):   # iterate through teses

        bay = tesid.split('_')[0]; row = tesid.split('_')[1]
        tlabels = [key for key in data[bay][row]['iv']]
        maxiv = max([len(data[bay][row]['iv'][tlab]['data'][0]) for tlab in tlabels])   # handle IVs of different lengths
        asize = (len(data[bay][row]['iv']), maxiv)   # temp length by maximum iv size
        vbias = np.full(asize, np.nan); vfb = np.full(asize, np.nan)   # initialize arrays
        meas_temps = np.array([np.nan]*len(tlabels)); rn_temp = np.array([np.nan]*len(meas_temps))
        v_pnts = np.zeros((len(tlabels), len(perRn))); i_pnts = np.zeros((len(tlabels), len(perRn))); p_pnts = np.zeros((len(tlabels), len(perRn)))   # initialize interpolated IVs

        
        for tt, tlab in enumerate(tlabels):   # iterate through temperatures
            meas_temps[tt] = data[bay][row]['iv'][tlab]['measured_temperature']
            ivlen = len(data[bay][row]['iv'][tlab]['data'][0])   # handle IVs of different lengths
            vbias[tt,:ivlen] = data[bay][row]['iv'][tlab]['data'][0,::-1]   # raw voltage, taken from high voltage -> 0
            vfb[tt,:ivlen] = data[bay][row]['iv'][tlab]['data'][1,::-1]   # raw current, taken from high voltage -> 0`

        plt.figure()
        labels = [str(round(mtemp*1e3)) + ' mK' for mtemp in meas_temps]
        plt.plot(vbias.T, vfb.T)
        plt.legend(labels)
        plt.title(tesid)
        plt.xlabel('Vbias')
        plt.ylabel('Vfb')
        plt.savefig(bolotest_dir + 'plots/Noisy_IVs/' + tesid + '_rawIVs.png', dpi=300)

if plot_mbolos:   # bolos missing from Shannon's spreadsheet

    mbolos = ['BayAX_Row00', 'BayAX_Row03', 'BayAX_Row06', 'BayAX_Row08', 'BayAX_Row10']

if save_pad18: # send P and T values for pad 18, bolo 20
    # with open(pkl_file, 'r') as pklfile:     
    #     ivs = pkl.load(pklfile)
    # pads = [int(ivs[tesid]['Pad']) for tesid in tesids]
    tesid = [tesid for tesid in tesids if ivs[tesid]['Pad']=='18'][0]
    fields = ivs[tesid]['meas_temps']
    # rows = [ivs[tesid]['ptes'], ivs[tesid]['meas_temps'], tes.sigma_power(ivs[tesid]['ites'], sigma_i, ivs[tesid]['vtes'], sigma_v)]
    rows1 = ivs[tesid]['ptes_fit'].T
    rows2 = ivs[tesid]['ptes_err'].T
    with open('/Users/angi/NIS/Analysis/bolotest/pad18_pvt.csv', 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)  # csv writer object  
        csvwriter.writerow(fields)  
        csvwriter.writerows(rows1)
        csvwriter.writerows(rows2)


if plot_bolo23:

    # bolo23 = 'BayAX_Row04'
    tesid = 'BayAX_Row04'   # just look at bolo 23
    temp_toplot = 0.170   # K


    # load analyzed data
    with open(pkl_file, "rb") as pf:
        ivs = pkl.load(pf)


    plotfit_singlebolo(ivs, tesid, save_figs=save_figs, fn_comments=fn_comments, bolotest_dir='/Users/angi/NIS/Bolotest_Analysis/')

    # data, alltesids = load_data(dfiles)   # load data for all TESs
    # ind23 = np.where(alltesids==bolo23)[0]; tesid = tesids[ind23][0]

    # ivs = convert_rawdata(data, tesids, constT=constT, v_nfit=v_nfit, save_figs=save_figs, iv_plots=iv_plots, branch_plots=branch_plots, tran_perRn_start=tran_perRn_start, perRn=perRn, 
    #                 fn_comments=fn_comments, rm_fl=remove_flagged, bolotest_dir=bolotest_dir)

    # ivs = fit_powerlaws(ivs, save_figs=save_figs, fitGexplicit=fitGexplicit, perRn_toquote=perRn_toquote, Tb_inds=Tb_inds, fn_comments=fn_comments, Tbq_ind=Tbq_ind, constT=constT, 
    #                   perRn=perRn, init_guess=init_guess, bolotest_dir=bolotest_dir)



if estimate_Gcov:   # plot G/n and G/k covariance 
    tesrange = np.arange(2)

    bays = ['BayAY', 'BayAX']

    ivs = {}   # master dictionary
    tesids = []
    data = {}
    for bb in np.arange(len(bays)):
        ### load data
        dfile = dfiles[bb]
        with open(dfile, "rb") as f:
            data_temp = pkl.load(f, encoding='latin1')
        tesids_temp = [(str(bays[bb])) + '_' + key for key in data_temp[bays[bb]]]
        if bays[bb] == 'BayAY':
            # remove noisy data (probably unlocked SQUID)
            tesids_temp.remove('BayAY_Row00')   # noisy
            tesids_temp.remove('BayAY_Row09')   # noisy
            tesids_temp.remove('BayAY_Row15')   # noisy
        elif bays[bb] == 'BayAX':  
            # remove noisy data (probably unlocked SQUID)
            tesids_temp.remove('BayAX_Row05')   
            tesids_temp.remove('BayAX_Row11')    
            tesids_temp.remove('BayAX_Row14')   
            tesids_temp.remove('BayAX_Row15') 
            pass
        data[bays[bb]] = data_temp[bays[bb]]
        tesids.extend(tesids_temp)

    # sort raw data and convert to real
    for ts, tesid in enumerate(tesids[0:3]):   # iterate through bolos

        bay = tesid.split('_')[0]; row = tesid.split('_')[1]
        if bay=='BayAY':
            pad = str(int(tesid.split('Row')[1])+1)  # pad number
        elif bay=='BayAX':
            pad = str(int(tesid.split('Row')[1])+16)  # pad number
        
        boloid = pad_bolo_map[pad]   # map TES ID to pad # to bolo ID

        tes = TESAnalyze() 
        ivs[tesid] = {}  
        ivs[tesid]['Pad'] = pad; ivs[tesid]['Bolometer'] = boloid   # save IDs to master dictionary
        tlabels = [key for key in data[bay][row]['iv']]
        if tesid == 'BayAY_Row12' or tesid == 'BayAX_Row10' or tesid == 'BayAX_Row03' or tesid == 'BayAX_Row00' or tesid == 'BayAX_Row06': 
            tlabels.remove('iv014')   # wonky IV
        maxiv = max([len(data[bay][row]['iv'][tlab]['data'][0]) for tlab in tlabels])   # handle IVs of different lengths
        asize = (len(data[bay][row]['iv']), maxiv)   # temp length by maximum iv size
        vbias = np.full(asize, np.nan); vfb = np.full(asize, np.nan)   # initialize arrays
        vtes = np.full(asize, np.nan); ites = np.full(asize, np.nan); rtes = np.full(asize, np.nan); i_meas = np.full(asize, np.nan); ptes = np.full(asize, np.nan)
        meas_temps = np.array([np.nan]*len(tlabels)); rn_temp = np.array([np.nan]*len(meas_temps))
        v_pnts = np.zeros((len(tlabels), len(perRn))); i_pnts = np.zeros((len(tlabels), len(perRn))); p_pnts = np.zeros((len(tlabels), len(perRn)))   # initialize interpolated IVs
        sigma_v = np.array([np.nan]*len(tlabels)); sigma_i = np.array([np.nan]*len(tlabels)); nfits_real = np.zeros((len(tlabels), 2))

        for tt, tlab in enumerate(tlabels):   # iterate through temperatures

            meas_temps[tt] = data[bay][row]['iv'][tlab]['measured_temperature']
            ivlen = len(data[bay][row]['iv'][tlab]['data'][0])   # handle IVs of different lengths
            vbias[tt,:ivlen] = data[bay][row]['iv'][tlab]['data'][0,::-1]   # raw voltage, taken from high voltage -> 0
            vfb[tt,:ivlen] = data[bay][row]['iv'][tlab]['data'][1,::-1]   # raw current, taken from high voltage -> 0
            vtes[tt], ites[tt], rtes[tt], ptes[tt], i_meas[tt], n_fit, norm_inds, sc_fit, end_sc, rpar = tes.ivAnalyzeTDM(vbias[tt], vfb[tt], Rfb, Rb, Rsh, M_r, v_nfit, show_plot=False)   # *
            nfits_real[tt] = np.polyfit(vtes[tt,norm_inds][0], ites[tt,norm_inds][0], 1)   # normal branch line fit in real IV units
            ifit_norm = vtes[tt,norm_inds]*nfits_real[tt, 0]+nfits_real[tt, 1]   # normal branch line fit
            rn_temp[tt] = np.mean(rtes[tt, norm_inds])   # ohms, should be consistent with 1/nfit_real[0]
            sigma_v[tt] = np.std(vtes[tt,:end_sc])   # V, error in voltage measurement = std of SC branch
            sigma_i[tt] = np.std(ites[tt,norm_inds] - ifit_norm)  # A, error in current measurement = std in normal branch after line subtraction
            v_pnts[tt], i_pnts[tt], p_pnts[tt] = tes.ivInterpolate(vtes[tt], ites[tt], rtes[tt], perRn, rn_temp[tt], tran_perRn_start=tran_perRn_start, plot=False)

        Tbaths = meas_temps[Tb_inds]
        if iv_plots: 
            plt.figure()
            tsort = np.argsort(meas_temps)
            for tt in tsort:
                plt.plot(v_pnts[tt]*1e6, i_pnts[tt]*1e3, 'o', alpha=0.7, color=plt.cm.plasma((meas_temps[tt]-min(Tbaths)*1)/(max(meas_temps))))   # meas_temps/max(meas_temps)
                plt.plot(vtes[tt]*1e6, ites[tt]*1e3, alpha=0.6, label='{} mK'.format(round(meas_temps[tt]*1E3,2)), color=plt.cm.plasma((meas_temps[tt]-min(Tbaths)*1)/(max(meas_temps)*0.8)))
            plt.xlabel('Voltage [$\mu$V]')
            plt.ylabel('Current [mA]')
            plt.title('Interpolated IV Points')
            plt.legend()
            if save_figs: plt.savefig(bolotest_dir + 'Plots/IVs/' + tesid + '_interpIVs' + fn_comments + '.png', dpi=300)

        rn = np.nanmean(rn_temp)
        rn_err = np.nanstd(rn_temp)

        Tb_toquote = meas_temps[Tbq_ind]   # measured temperature to quote
        qind = np.where(perRn==perRn_toquote)[0][0]
        tind = np.where(Tbaths==Tb_toquote)[0][0]  # index of subset Tbaths to return
        sigma_p = np.zeros(np.shape(i_pnts))   # this is stupid but it works
        for ii, ipnt in enumerate(i_pnts):
            sigma_p[ii] = tes.sigma_power(i_pnts[ii], sigma_i[ii], v_pnts[ii], sigma_v[ii])

        # get initial fit for setting k and n ranges
        params_init = tes.fitPowerLaw(perRn, meas_temps, p_pnts.T, init_guess, fitToLast=True, 
                suptitle=tesid, Tbaths=Tbaths, plot=False, sigma=sigma_p.T, constT=True, fitGexplicit=False)     
        GTc0 = params_init[0][qind]; sigma_GTc0 = params_init[4][qind]
        K0 = params_init[1][qind]; sigma_K0 = params_init[5][qind]
        n0 = params_init[2][qind]; sigma_n0 = params_init[6][qind]

        params_fitG = tes.fitPowerLaw(perRn, meas_temps, p_pnts.T, init_guess, fitToLast=True, 
                suptitle=tesid, Tbaths=Tbaths, plot=False, sigma=sigma_p.T, constT=True, fitGexplicit=True)    
        GTc0_fitG = params_fitG[0][qind]; sigma_GTc0_fitG = params_fitG[4][qind]

        K_range = np.linspace(K0 - sigma_K0, K0 + sigma_K0)
        GTc_krange = np.zeros(len(K_range)); n_krange = np.zeros(len(K_range))
        init_guess_k = init_guess[[1,2]] 

        for kk, Kv in enumerate(K_range): 
            params_temp = tes.fitPowerLaw(perRn, meas_temps, p_pnts.T, init_guess_k, fitToLast=True, 
                suptitle=tesid, Tbaths=Tbaths, plot=False, sigma=sigma_p.T, constT=True, fitGexplicit=False, fixedK=Kv) 
            GTc_krange[kk] = params_temp[0][qind]; n_krange[kk] = params_temp[2][qind]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(K_range*1E11, n_krange, alpha=.5, color='C0', marker='o', markeredgewidth=1.0)
        ax2.plot(K_range*1E11, GTc_krange*1E12, alpha=.5, color='C3', marker='o')
        plt.errorbar(K0*1E11, GTc0_fitG*1E12, yerr=sigma_GTc0_fitG*1E12, color='k', marker='x', capsize=4, alpha=0.7, label='Full G Fit Result')
        ax1.set_xlabel('k [1E11]')
        ax1.set_ylabel('n', color='C0'); ax2.set_ylabel('GTc [pW/K]', color='C3')
        ax1.tick_params(axis='y', labelcolor='C0'); ax2.tick_params(axis='y', labelcolor='C3')
        plt.legend()
        plt.title('Varying k - Pad ' + pad)
        
        plt.figure()
        plt.plot(K_range*1E11, GTc_krange*1E12, alpha=.5, color='C3', marker='o')
        plt.errorbar(K0*1E11, GTc0*1E12, yerr=sigma_GTc0*1E12, color='C3', marker='x', capsize=4, alpha=0.7, label='Full k Fit Result')
        plt.errorbar(K0*1E11, GTc0_fitG*1E12, yerr=sigma_GTc0_fitG*1E12, color='k', marker='x', capsize=4, alpha=0.7, label='Full G Fit Result')
        plt.xlabel('k [1E11]'); plt.ylabel('GTc [pW/K]')
        plt.legend()
        plt.title('Varying k - Pad ' + pad)

        # nval = params_init[2][qind]; sigma_n = params_init[6][qind]
        n_range = np.linspace(n0 - sigma_n0, n0 + sigma_n0)
        GTc_nrange = np.zeros(len(n_range)); K_nrange = np.zeros(len(n_range))
        init_guess_n = init_guess[[0,2]]
        for nn, nv in enumerate(n_range):
            params_temp = tes.fitPowerLaw(perRn, meas_temps, p_pnts.T, init_guess_n, fitToLast=True, 
                suptitle=tesid, Tbaths=Tbaths, plot=False, sigma=sigma_p.T, constT=True, fitGexplicit=False, fixedn=nv) 
            GTc_nrange[nn] = params_temp[0][qind]; K_nrange[nn] = params_temp[1][qind]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(n_range, K_nrange*1E11, alpha=.5, color='C1', marker='o', markeredgewidth=1.0)
        # ax1.errorbar(n0, K0*1E11, yerr=sigma_K0*1E11, color='C0', marker='x', capsize=4, alpha=0.5, label='Full k Fit Result - k')
        ax2.plot(n_range, GTc_nrange*1E12, alpha=.5, color='C3', marker='o')
        plt.errorbar(n0, GTc0_fitG*1E12, yerr=sigma_GTc0_fitG*1E12, color='k', marker='x', capsize=4, alpha=0.7, label='Full G Fit Result')
        ax1.set_xlabel('n')
        ax1.set_ylabel('k [1E11]', color='C1'); ax2.set_ylabel('GTc [pW/K]', color='C3')
        ax1.tick_params(axis='y', labelcolor='C1'); ax2.tick_params(axis='y', labelcolor='C3')
        plt.legend()
        plt.title('Varying n - Pad ' + pad)
        
        plt.figure()
        plt.plot(n_range, GTc_nrange*1E12, alpha=.5, color='C3', marker='o')
        plt.errorbar(n0, GTc0*1E12, yerr=sigma_GTc0*1E12, color='C3', marker='x', capsize=4, alpha=0.7, label='Full k Fit Result')
        plt.errorbar(n0, GTc0_fitG*1E12, yerr=sigma_GTc0_fitG*1E12, color='k', marker='x', capsize=4, alpha=0.7, label='Full G Fit Result')
        plt.xlabel('n'); plt.ylabel('GTc [pW/K]')
        plt.legend()
        plt.title('Varying n - Pad ' + pad)

if show_figs: plt.show() 
