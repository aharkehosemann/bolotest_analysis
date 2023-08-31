""" 
Reanalyze bolotest data from 2018 (?)
AHHH 2020/12

to do: fix SC branch of IVs; second y axis on Psat plot
automated error analysis: baseline removal tes-by-tes basis scipy.signal.sabgal_filter
"""
import numpy as np 
import matplotlib.pyplot as plt
import pickle as pkl
import tesanalyze_ahhh
import pdb
import csv 
from collections import OrderedDict

### user params
# which analysis?
analyze_ivs = True
estimate_Gcov = False   # estimating n, k, G covariance to inform error analysis
plot_noisyivs = False
plot_mbolos = False
save_pad18 = False   # write quick CSV for Joel
scaleG = False   # scale G(Tc) to G(170 mK)
plot_bolo23 = False

# analysis options
save_data = True   # overwrite csv and pkl data files
save_figs = True  # overwrite figures
latex_fonts = False
constT = True   # save constant T_TES assumption results
fitGexplicit = True   # fit G explicitly
show_ivplots = False   # show plots converting raw IV to real IV
show_aplots = True   # helpful for double-checking IV analysis
show_psatcalc = False   # i've never seen the Psat calc not work
plot_byind = False   # for easier IV exclusion 

bolotest_dir = '/Users/angi/NIS/Bolotest_Analysis/'
# fn_comments = '_fitGexplicit_constTTES_rerun'   # fit G explicitly, assume constant T_TES regardless of %Rn
fn_comments = '_reanalyzed_SCbranch'   # fit G explicitly, assume constant T_TES regardless of %Rn
dfiles = ['/Users/angi/NIS/Bolotest_Analysis/Data/MM2017L_20171117_data/AY_1thru15_IVvsTb.pkl', '/Users/angi/NIS/Bolotest_Analysis/Data/MM2017L_20171117_data/AX_16thru30_IVvsTb.pkl']
csv_file = bolotest_dir + 'Analysis_Files/bolotest_reanalyzedAHHH_202308' + fn_comments + '.csv'   # where to save analysis results
pkl_file = bolotest_dir + 'Analysis_Files/bolotest_reanalyzedAHHH_202308' + fn_comments + '.pkl'   # where to save analysis results

# pRn = np.array([25, 30, 40, 50, 60, 70, 80, 90])   # % Rn
pRn = np.array([25, 30, 40, 50, 60, 70, 80])   # % Rn
v_nfit = .3   # v_bias above which TES is normal (approximate)
tran_pRn_start = 0.2   # Fraction of Rn dubbed beginning of SC transition
init_guess = np.array([1.E-10, 2.5, .170]) if not fitGexplicit else np.array([10E-12, 2.5, .170])  # kappa, n, Tc [mK]; powerlaw fitter
# v_offset = 0   # V; SC and/or normal branch should go through (0,0)
# i_offset = 0.*1e-6   # Amps; SC and/or normal branch should go through (0,0)
tinds_return = np.array([0, 3, 8, 11, 15, 2])
# Tb_ind = 0   # Tbath index for Tbath at which to calculate Psat and G(Tc) (150 mK)
# Tb_ind = 15   # Tbath index for Tbath at which to calculate Psat and G(Tc) (170 mK)
Tb_ind = 2   # Tbath index for Tbath at which to calculate Psat and G(Tc) (168 mK); there isn't always a Tbath = 165 mK measurement
pRn_toquote = 80   # % Rn fit values to print to screen

# readout circuit parameters
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
    for ts, tesid in enumerate(tesids):   # iterate through bolos

        bay = tesid.split('_')[0]; row = tesid.split('_')[1]
        if bay=='BayAY':
            pad = str(int(tesid.split('Row')[1])+1)  # pad number
        elif bay=='BayAX':
            pad = str(int(tesid.split('Row')[1])+16)  # pad number
        boloid = pad_bolo_map[pad]   # map TES ID to pad # to bolo ID

        tes = tesanalyze_ahhh.TESAnalyze() 
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
        v_pnts = np.zeros((len(tlabels), len(pRn))); i_pnts = np.zeros((len(tlabels), len(pRn))); p_pnts = np.zeros((len(tlabels), len(pRn)))   # initialize interpolated IVs
        sigma_v = np.array([np.nan]*len(tlabels)); sigma_i = np.array([np.nan]*len(tlabels)); nfits_real = np.zeros((len(tlabels), 2))

        for tt, tlab in enumerate(tlabels):   # iterate through temperatures
            meas_temps[tt] = data[bay][row]['iv'][tlab]['measured_temperature']
            ivlen = len(data[bay][row]['iv'][tlab]['data'][0])   # handle IVs of different lengths
            vbias[tt,:ivlen] = data[bay][row]['iv'][tlab]['data'][0,::-1]   # raw voltage, taken from high voltage -> 0
            vfb[tt,:ivlen] = data[bay][row]['iv'][tlab]['data'][1,::-1]   # raw current, taken from high voltage -> 0
            vtes[tt], ites[tt], rtes[tt], ptes[tt], i_meas[tt], n_fit, norm_inds, sc_fit, end_sc, rpar = tes.ivAnalyzeTDM(vbias[tt], vfb[tt], Rfb, Rb, Rsh, M_r, v_nfit, show_plot=show_ivplots)
            if show_ivplots: plt.title('Bolo {boloid}, Pad {pad} - Raw IV at {tlab} mK'.format(boloid=boloid, pad=pad, tlab=round(meas_temps[tt]*1E3)))

            # pdb.set_trace()
            finds = np.isfinite(ites[tt])   # ignore nans
            nfinds = np.nonzero(np.in1d(finds, norm_inds))[0]   # normal and finite data points
            nfits_real[tt] = np.polyfit(vtes[tt,norm_inds][0], ites[tt,norm_inds][0], 1)   # normal branch line fit in real IV units
            ifit_norm = vtes[tt,norm_inds]*nfits_real[tt, 0]+nfits_real[tt, 1]   # normal branch line fit
            rn_temp[tt] = np.mean(rtes[tt, norm_inds])   # ohms, should be consistent with 1/nfit_real[0]
            sigma_v[tt] = np.std(vtes[tt,:end_sc-2])   # V, error in voltage measurement = std of SC branch
            sigma_i[tt] = np.std(ites[tt,norm_inds] - ifit_norm)  # A, error in current measurement = std in normal branch after line subtraction
            v_pnts[tt], i_pnts[tt], p_pnts[tt] = tes.ivInterpolate(vtes[tt], ites[tt], rtes[tt], pRn, rn_temp[tt], tran_pRn_start=tran_pRn_start, plot=False)
        
        TbsToReturn = meas_temps[tinds_return]
        # pdb.set_trace()
        if show_aplots: 
            plt.figure()
            tsort = np.argsort(meas_temps)
            for tt in tsort:
                finds = np.isfinite(ites[tt])
                plt.plot(v_pnts[tt]*1e6, i_pnts[tt]*1e3, 'o', alpha=0.7, color=plt.cm.plasma((meas_temps[tt]-0.070)/(0.170)))   # meas_temps/max(meas_temps)
                plt.plot(vtes[tt][finds]*1e6, ites[tt][finds]*1e3, alpha=0.6, label='{} mK'.format(round(meas_temps[tt]*1E3,0)), color=plt.cm.plasma((meas_temps[tt]-0.070)/(0.170)))
            plt.xlabel('Voltage [$\mu$V]')
            plt.ylabel('Current [mA]')
            plt.title('Interpolated IV Points - Bolo {boloid}, Pad {pad}'.format(boloid=boloid, pad=pad))
            plt.legend()
            if save_figs: plt.savefig(bolotest_dir + 'Plots/IVs/pad' + pad + '_interpIVs' + fn_comments + '.png', dpi=300)
            # plt.show()
            # pdb.set_trace()

        rn = np.nanmean(rn_temp)
        rn_err = np.nanstd(rn_temp)

        Tb_toquote = meas_temps[Tb_ind]
        qind = np.where(pRn==pRn_toquote)[0][0]
        tind = np.where(TbsToReturn==Tb_toquote)[0][0]  
        sigma_p = np.zeros(np.shape(i_pnts))   # this is stupid but it works
        for ii, ipnt in enumerate(i_pnts):
            sigma_p[ii] = tes.sigma_power(i_pnts[ii], sigma_i[ii], v_pnts[ii], sigma_v[ii])
        # pdb.set_trace()

        ### fit power law
        pfig_path = bolotest_dir + 'Plots/Psat_fits/' + tesid + '_Pfit' + fn_comments + '.png' if save_figs else None
        GTcs, Ks, ns, Ttes, GTcs_err, Ks_err, ns_err, Ttes_err = tes.fitPowerLaw(pRn, meas_temps, p_pnts.T, init_guess, fitToLast=True, 
                suptitle=tesid, TbsToReturn=TbsToReturn, plot=True, sigma=sigma_p.T, nstd=5, pfigpath=pfig_path, constT=constT, fitGexplicit=fitGexplicit)   # pass error to fitter     
        if save_figs: plt.savefig(bolotest_dir + 'Plots/fit_params/' + tesid + '_fitparams' + fn_comments + '.png', dpi=300) 
        
        if constT:
            GTc_toquote = GTcs[qind]; GTcerr_toquote = GTcs_err[qind]
            Tc_toquote = Ttes[qind]; Tcerr_toquote = Ttes_err[qind]
        else:
            GTc_toquote = GTcs[tind, qind]; GTcerr_toquote = GTcs_err[tind, qind]
            Tc_toquote = Ttes[tind, qind]; Tcerr_toquote = Ttes_err[tind, qind]

        print(' ')
        print(' ')
        print(tesid)
        print('G = ', round(GTc_toquote*1e12, 2), ' +/- ', round(GTcerr_toquote*1e12, 2), 'pW/K')
        print('K = ',  round(Ks[qind]*1E11, 3), ' +/- ',  round(Ks_err[qind]*1E11, 3), ' E-11') 
        print('n = ', round(ns[qind], 2), ' +/- ', round(ns_err[qind], 4))
        print('Tc = ', round(Tc_toquote*1e3, 2), ' +/- ',  round(Tcerr_toquote*1e3, 2), 'mK')
        print('TES Rn = ', round(rn*1e3, 2), ' +/- ', round(rn_err*1e3, 2), ' mOhms')

        ### calculate Psat
        # find transition + normal branch
        sc_inds = np.where((rtes[Tb_ind]/rn)<.2)[0]
        start_ind = np.max(sc_inds)
        end_ind = np.max(np.where(((rtes[Tb_ind]/rn)>.2) & (rtes[Tb_ind]!=np.nan)))
        vtes_tran = vtes[Tb_ind, start_ind:end_ind]
        ites_tran = ites[Tb_ind, start_ind:end_ind]
        rtes_tran = rtes[Tb_ind, start_ind:end_ind]

        # calculate Psat
        ptes_tran = vtes_tran * ites_tran
        sat_ind = np.where(ites_tran == np.min(ites_tran))[0][0]   # where the TES goes normal
        Psat = ptes_tran[sat_ind]
        Psat_err = tes.sigma_power(ites_tran[sat_ind], sigma_i[Tb_ind], vtes_tran[sat_ind], sigma_v[Tb_ind])
        Psat_calc = tes.Psat_atT(Tb_toquote, Tc_toquote, Ks[qind], ns[qind])
        print('Psat = ', round(Psat*1e12, 4), ' +/- ', round(Psat_err*1e12, 4), 'pW')
        print('Psat (calc) = ', round(Psat_calc*1e12, 4), 'pW')
        print(' ')
        print(' ')
        if show_psatcalc:
            plt.figure()
            plt.plot(vtes_tran.T*1e6, ites_tran.T/np.max(ites_tran), label='TES IV')
            plt.plot(vtes_tran.T*1e6, ptes_tran.T/np.max(ptes_tran), label='Power')
            plt.plot(vtes_tran[sat_ind]*1e6, Psat/np.max(ptes_tran), 'x', label='$P_{sat}$')
            plt.xlabel('Voltage [$\mu$V]')
            plt.ylabel('Normalized Current')
            plt.legend()
            plt.title('TES IV and Calculated Power at Tbath = ' + str(round(Tb_toquote*1000, 1)) + 'mK')
            if save_figs: plt.savefig(bolotest_dir + 'Plots/psat_calc/' + tesid + '_psatcalc' + fn_comments + '.png', dpi=300)

        # store results in dict
        sort_inds = np.argsort(meas_temps)   # sort by temp, ignore nans
        ivs[tesid]['TES ID'] = tesid  
        ivs[tesid]['meas_temps'] = meas_temps[sort_inds]   # K
        ivs[tesid]['constT'] = constT
        ivs[tesid]['vbias'] = vbias[sort_inds]  
        ivs[tesid]['vfb'] = vfb[sort_inds]  
        ivs[tesid]['vtes'] = vtes[sort_inds]   # volts
        ivs[tesid]['ites'] = ites[sort_inds]   # amps
        ivs[tesid]['rtes'] = rtes[sort_inds]   # ohms
        ivs[tesid]['ptes'] = ptes[sort_inds]   # power
        ivs[tesid]['ptes_fit'] = p_pnts
        ivs[tesid]['ptes_err'] = sigma_p
        ivs[tesid]['i_meas'] = i_meas[sort_inds]   # amps
        ivs[tesid]['Rn [mOhms]'] = rn*1e3   # ohms
        ivs[tesid]['Rn_err [mOhms]'] = rn_err*1e3   # ohms
        ivs[tesid]['fitGexplicit'] = fitGexplicit
        ivs[tesid]['G@Tc [pW/K]'] = GTc_toquote*1e12 
        ivs[tesid]['G_err@Tc [pW/K]'] = GTcerr_toquote*1e12
        ivs[tesid]['G@170mK [pW/K]'] = tes.scale_G(.170, GTc_toquote, Tc_toquote, ns[qind])*1e12  
        ivs[tesid]['G_err@170mK [pW/K]'] = tes.sigma_GscaledT(.170, GTc_toquote, Tc_toquote, ns[qind], GTcerr_toquote, Tcerr_toquote, ns_err[qind])*1e12  
        ivs[tesid]['k'] = Ks[qind] 
        ivs[tesid]['k_err'] = Ks_err[qind] 
        ivs[tesid]['n'] = ns[qind] 
        ivs[tesid]['n_err'] = ns_err[qind] 
        ivs[tesid]['Tc [mK]'] = Tc_toquote*1e3
        ivs[tesid]['Tc_err [mK]'] = Tcerr_toquote*1e3
        ivs[tesid]['Psat@'+str(round(Tb_toquote*1e3))+'mK [pW], IV'] =  Psat*1e12
        ivs[tesid]['Psat_err@'+str(round(Tb_toquote*1e3))+'mK [pW], IV'] =  Psat_err*1e12
        ivs[tesid]['Psat@'+str(round(Tb_toquote*1e3))+'mK [pW], Calc'] =  Psat_calc*1e12

    if save_data:
        # write CSV     
        pads = [int(ivs[tesid]['Pad']) for tesid in tesids]
        fields = np.array(['TES ID', 'Bolometer', 'Pad', 'Tc [mK]', 'Tc_err [mK]', 'Rn [mOhms]', 'Rn_err [mOhms]', 'G@Tc [pW/K]', 'G_err@Tc [pW/K]', 'G@170mK [pW/K]', 'G_err@170mK [pW/K]', 'k', 'k_err', 'n', 'n_err', 'Psat@'+str(round(Tb_toquote*1e3))+'mK [pW], IV', 'Psat_err@'+str(round(Tb_toquote*1e3))+'mK [pW], IV', 'Psat@'+str(round(Tb_toquote*1e3))+'mK [pW], Calc'])  
        rows = np.array([[ivs[tesids[pp]][field] for field in fields] for pp in np.argsort(pads)])
        with open(csv_file, 'w') as csvfile:  
            csvwriter = csv.writer(csvfile)  # csv writer object  
            csvwriter.writerow(fields)  
            csvwriter.writerows(rows)

        # write pickle
        with open(pkl_file, 'wb') as pklfile:
            pkl.dump(ivs, pklfile)


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
        v_pnts = np.zeros((len(tlabels), len(pRn))); i_pnts = np.zeros((len(tlabels), len(pRn))); p_pnts = np.zeros((len(tlabels), len(pRn)))   # initialize interpolated IVs

        
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
    ivs = {} 
    # tesids = []
    data = {}
    for bb in np.arange(len(bays)):
        ### load data
        dfile = dfiles[bb]
        with open(dfile, "rb") as f:
            data_temp = pkl.load(f, encoding='latin1')
        data[bays[bb]] = data_temp[bays[bb]]

    for ts, tesid in enumerate(mbolos):   # iterate through teses
        bay = tesid.split('_')[0]; row = tesid.split('_')[1]
        tes = tesanalyze_ahhh.TESAnalyze() 
        ivs[tesid] = {}  
        tlabels = [key for key in data[bay][row]['iv']]
        if tesid == 'BayAY_Row12': 
            tlabels.remove(tlabels[15])   # wonky IV
        if bay == 'BayAX':
            tlabels.remove(tlabels[15])   # many of these are almost normal and confuse the fitter
        maxiv = max([len(data[bay][row]['iv'][tlab]['data'][0]) for tlab in tlabels])   # handle IVs of different lengths
        asize = (len(data[bay][row]['iv']), maxiv)   # temp length by maximum iv size
        vbias = np.full(asize, np.nan); vfb = np.full(asize, np.nan)   # initialize arrays
        vtes = np.full(asize, np.nan); ites = np.full(asize, np.nan); rtes = np.full(asize, np.nan); i_meas = np.full(asize, np.nan); ptes = np.full(asize, np.nan)
        meas_temps = np.array([np.nan]*len(tlabels)); rn_temp = np.array([np.nan]*len(meas_temps))
        v_pnts = np.zeros((len(tlabels), len(pRn))); i_pnts = np.zeros((len(tlabels), len(pRn))); p_pnts = np.zeros((len(tlabels), len(pRn)))   # initialize interpolated IVs
        
        plt.figure()
        for tt, tlab in enumerate(tlabels):   # iterate through temperatures

            meas_temps[tt] = data[bay][row]['iv'][tlab]['measured_temperature']
            ivlen = len(data[bay][row]['iv'][tlab]['data'][0])   # handle IVs of different lengths
            vbias[tt,:ivlen] = data[bay][row]['iv'][tlab]['data'][0,::-1]   # raw voltage, taken from high voltage -> 0
            vfb[tt,:ivlen] = data[bay][row]['iv'][tlab]['data'][1,::-1]   # raw current, taken from high voltage -> 0
            vtes[tt], ites[tt], rtes[tt], ptes[tt], i_meas[tt], n_fit, sc_fit, rpar = tes.ivAnalyzeTDM(vbias[tt], vfb[tt], Rfb, Rb, Rsh, M_r, v_nfit, show_plot=show_ivplots)
            # norm_ind = np.where(vbias[tt] > v_nfit)
            rn_temp[tt] = np.mean(rtes[tt, np.where(vbias[tt] > v_nfit)])   # ohms

            # interpolate IV points    
            lab=tt if (plot_byind) else str(round(meas_temps[tt]*1e3)) + ' mK'
            plt.plot(vtes[tt]*1e6, ites[tt]*1e3, label=lab, alpha=0.7)
            plt.legend()
            v_pnts[tt], i_pnts[tt], p_pnts[tt] = tes.ivInterpolate(vtes[tt], ites[tt], rtes[tt], pRn, rn_temp[tt], tran_pRn_start=tran_pRn_start)
        plt.xlabel('Voltage [$\mu$V]')
        plt.ylabel('Current [mA]')
        plt.title('Interpolated IV Points, ' + tesid)
        plt.savefig(bolotest_dir + 'plots/Missing_IVs/' + tesid + '_interpIVs.png', dpi=300)

        plt.figure()
        labels = [str(round(mtemp*1e3)) + ' mK' for mtemp in meas_temps]
        plt.plot(vbias.T, vfb.T)
        plt.legend(labels)
        plt.title(tesid)
        plt.xlabel('Vbias')
        plt.ylabel('Vfb')
        plt.savefig(bolotest_dir + 'plots/Missing_IVs/' + tesid + '_rawIVs.png', dpi=300)

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

if scaleG:

    Tcs = np.array([170.9877185, 170.9706609, 171.7552114, 171.7208656, 171.2176826, 170.9077816, 171.3470285, 171.0853375, 170.9200613, 171.060197, 170.0255488, 171.021025, 171.2850107, 170.4034122, 171.2042771, 170.2338939, 170.6389945, 172.0684101, 170.3518396, 171.721552, 171.5840882, 171.1586009, 170.8315709, 172.2051357, 171.7151257])
    sigma_Tcs = np.array([0.010421243, 0.009226958, 0.010790079, 0.009281995, 0.006323179, 0.009746988, 0.010994352, 0.009444727, 0.007650692, 0.005877343, 0.017378697, 0.009453003, 0.009177113, 0.021876877, 0.006768265, 0.006364612, 0.018369732, 0.012367842, 0.020356169, 0.007980188, 0.013312168, 0.006793048, 0.014202084, 0.009438768, 0.009855968])
    ns = np.array([2.102652493, 2.286305877, 2.181787081, 2.636751655, 3.625212281, 2.065019714, 2.030684323, 2.310936272, 2.271596966, 3.713230268, 2.05911454, 2.183641919, 2.389944922, 1.892766183, 3.29595625, 2.328520321, 2.105820885, 2.208529433, 1.994745782, 3.402832364, 2.135441834, 3.025155869, 2.821070825, 3.025157842, 2.816065888])
    sigma_ns = np.array([2.102652493, 2.286305877, 2.181787081, 2.636751655, 3.625212281, 2.065019714, 2.030684323, 2.310936272, 2.271596966, 3.713230268, 2.05911454, 2.183641919, 2.389944922, 1.892766183, 3.29595625, 2.328520321, 2.105820885, 2.208529433, 1.994745782, 3.402832364, 2.135441834, 3.025155869, 2.821070825, 3.025157842, 2.816065888])
    GTcs = np.array([7.063630682, 8.23553356, 10.31889778, 13.74098652, 21.98669484, 6.659975444, 7.601344581, 8.013653731, 10.63671194, 22.09412114, 4.67333846, 7.904637521, 10.03001622, 3.471988312, 16.47703326, 5.184876564, 5.196919583, 8.268319188, 3.703848367, 16.79390843, 5.659883891, 15.11216746, 10.07493674, 15.49026988, 10.44634148])
    sigma_GTcs = np.array([0.102018534, 0.112406438, 0.131803919, 0.171320173, 0.239205548, 0.092635138, 0.104237744, 0.1090491, 0.124941174, 0.238423651, 0.08774709, 0.108691126, 0.130040288, 0.072825133, 0.198879142, 0.078156549, 0.092319898, 0.113998559, 0.074520238, 0.204577595, 0.090894549, 0.167780174, 0.164340968, 0.18151003, 0.147841826])

    Tscale = .170
    scaledGs = tes.scale_G(Tscale, GTcs, Tcs/1E3, ns)  # pW/K
    sigma_scaledGs = tes.sigma_GscaledT(Tscale, GTcs, Tcs/1E3, ns, sigma_GTcs, sigma_Tcs/1E3, sigma_ns)

if plot_bolo23:

    bolo23 = 'BayAX_Row04'
    temp_toplot = 0.170   # K

    tes = tesanalyze_ahhh.TESAnalyze() 

    data, tesids = tes.load_data(dfiles)

    ind23 = np.where(tesids==bolo23)[0]
    tesid = tesids[ind23][0]

    ivs = tes.convert_rawdata(data, np.array([tesid]))

    # # sort raw data and convert to real
    # bay = tesid.split('_')[0]; row = tesid.split('_')[1]
    # if bay=='BayAY':
    #     pad = str(int(tesid.split('Row')[1])+1)  # pad number
    # elif bay=='BayAX':
    #     pad = str(int(tesid.split('Row')[1])+16)  # pad number
    
    # ivs = {}   # main dict

    # boloid = pad_bolo_map[pad]   # map TES ID to pad # to bolo ID

    # ivs[tesid] = {}  
    # ivs[tesid]['Pad'] = pad; ivs[tesid]['Bolometer'] = boloid   # save IDs to master dictionary
    # tlabels = [key for key in data[bay][row]['iv']]
    # if tesid == 'BayAY_Row12' or tesid == 'BayAX_Row10' or tesid == 'BayAX_Row03' or tesid == 'BayAX_Row00' or tesid == 'BayAX_Row06': 
    #     tlabels.remove('iv014')   # wonky IV
    # maxiv = max([len(data[bay][row]['iv'][tlab]['data'][0]) for tlab in tlabels])   # handle IVs of different lengths
    # asize = (len(data[bay][row]['iv']), maxiv)   # temp length by maximum iv size
    # vbias = np.full(asize, np.nan); vfb = np.full(asize, np.nan)   # initialize arrays
    # vtes = np.full(asize, np.nan); ites = np.full(asize, np.nan); rtes = np.full(asize, np.nan); i_meas = np.full(asize, np.nan); ptes = np.full(asize, np.nan)
    # meas_temps = np.array([np.nan]*len(tlabels)); rn_temp = np.array([np.nan]*len(meas_temps))
    # v_pnts = np.zeros((len(tlabels), len(pRn))); i_pnts = np.zeros((len(tlabels), len(pRn))); p_pnts = np.zeros((len(tlabels), len(pRn)))   # initialize interpolated IVs
    # sigma_v = np.array([np.nan]*len(tlabels)); sigma_i = np.array([np.nan]*len(tlabels)); nfits_real = np.zeros((len(tlabels), 2))

    
    # for tt, tlab in enumerate(tlabels):   # iterate through temperatures

    #     meas_temps[tt] = data[bay][row]['iv'][tlab]['measured_temperature']
    #     ivlen = len(data[bay][row]['iv'][tlab]['data'][0])   # handle IVs of different lengths
    #     vbias[tt,:ivlen] = data[bay][row]['iv'][tlab]['data'][0,::-1]   # raw voltage, taken from high voltage -> 0
    #     vfb[tt,:ivlen] = data[bay][row]['iv'][tlab]['data'][1,::-1]   # raw current, taken from high voltage -> 0
    #     vtes[tt], ites[tt], rtes[tt], ptes[tt], i_meas[tt], n_fit, norm_inds, sc_fit, end_sc, rpar = tes.ivAnalyzeTDM(vbias[tt], vfb[tt], Rfb, Rb, Rsh, M_r, v_nfit, show_plot=False)   # *
    #     nfits_real[tt] = np.polyfit(vtes[tt,norm_inds][0], ites[tt,norm_inds][0], 1)   # normal branch line fit in real IV units
    #     ifit_norm = vtes[tt,norm_inds]*nfits_real[tt, 0]+nfits_real[tt, 1]   # normal branch line fit
    #     rn_temp[tt] = np.mean(rtes[tt, norm_inds])   # ohms, should be consistent with 1/nfit_real[0]
    #     sigma_v[tt] = np.std(vtes[tt,:end_sc])   # V, error in voltage measurement = std of SC branch
    #     # sigma_i[tt] = np.std(ites[tt,norm_inds] - ifit_norm)  # A, error in current measurement = std in normal branch after line subtraction
    #     sigma_i[tt] = np.std(ites[tt,norm_inds])  # A, error in current measurement = std in normal branch after line subtraction

    #     v_pnts[tt], i_pnts[tt], p_pnts[tt] = tes.ivInterpolate(vtes[tt], ites[tt], rtes[tt], pRn, rn_temp[tt], tran_pRn_start=tran_pRn_start, plot=False)


    # if show_aplots: 
    #     plt.figure()
    #     tsort = np.argsort(meas_temps)
    #     for tt in tsort:
    #         plt.plot(v_pnts[tt]*1e6, i_pnts[tt]*1e3, 'o', color=plt.cm.plasma(meas_temps[tt]/(max(meas_temps)*1.3)))   # meas_temps/max(meas_temps)
    #         plt.plot(vtes[tt]*1e6, ites[tt]*1e3, alpha=0.7, label='{} mK'.format(round(meas_temps[tt]*1E3,2)), color=plt.cm.plasma(meas_temps[tt]/(max(meas_temps)*1.3)))
    #     plt.xlabel('Voltage [$\mu$V]')
    #     plt.ylabel('Current [mA]')
    #     plt.title('Interpolated IV Points')
    #     # handles, labels = plt.gca().get_legend_handles_labels()
    #     # by_label = OrderedDict(zip(labels, handles))
    #     # plt.legend(by_label.values(), by_label.keys())
    #     plt.legend()
    #     if save_figs: plt.savefig(bolotest_dir + 'Plots/for_paper/bolo23_interpIVs' + fn_comments + '_forpaper.png', dpi=300)

    # mintemp = 0.150
    # # tinds = np.where(meas_temps>mintemp)[0]
    # # ctemps = meas_temps-mintemp
    # colors = plt.cm.plasma([(temp/(max(meas_temps)-mintemp))*0.9 for temp in meas_temps])
    # # labs = np.array([(str(np.round(temp*1E3))+' mK') for temp in meas_temps])
    # plt.figure()
    # for tt in tsort:
    #     if meas_temps[tt]>=mintemp:
    #         plt.plot(vtes[tt].T*1e9, ites[tt].T*1e6, alpha=0.7, color=plt.cm.plasma((meas_temps[tt]-mintemp)/(max(meas_temps)-mintemp)*0.9), label='{:.0f} mK'.format(round(meas_temps[tt]*1E3,2)))
    # plt.xlabel('Voltage [nV]')
    # plt.ylabel('Current [$\mu$A]')    
    # # handles, labels = plt.gca().get_legend_handles_labels()   # deal with redundant labels
    # # slinds = np.argsort(labs[tinds])   # sort labels
    # # plt.legend([handles[ind] for ind in slinds],[labels[ind] for ind in slinds])
    # plt.legend()
    # # plt.title('Bolo 3 I-V Characteristics')
    # plt.xlim(-15, 200)
    # if save_figs: plt.savefig(bolotest_dir + 'Plots/for_paper/bolo23_IVs' + fn_comments + '_forpaper.png', dpi=300) 

    # # tind = np.where((meas_temps<temp_toplot+0.001) & (meas_temps>temp_toplot-0.001))[0][0]
    # # plt.figure()
    # # plt.plot(vtes[tind]*1e9, ites[tind]*1e6, alpha=0.7)
    # # plt.xlabel('Voltage [nV]')
    # # plt.ylabel('Current [$\mu$A]')
    # # plt.title('Bolo 23 IV at 170 mK')
    # # if save_figs: plt.savefig(bolotest_dir + 'Plots/for_paper/bolo23_IV' + fn_comments + '_forpaper.png', dpi=300) 

    # rn = np.nanmean(rn_temp)
    # rn_err = np.nanstd(rn_temp)

    # TbsToReturn = meas_temps[tinds_return]
    # Tb_toquote = meas_temps[Tb_ind]
    # qind = np.where(pRn==pRn_toquote)[0][0]
    # tind = np.where(TbsToReturn==Tb_toquote)[0][0]
    # sigma_p = np.zeros(np.shape(i_pnts))   # this is stupid but it works
    # for ii, ipnt in enumerate(i_pnts):
    #     sigma_p[ii] = tes.sigma_power(i_pnts[ii], sigma_i[ii], v_pnts[ii], sigma_v[ii])

    # ### fit power law
    # pfig_path = bolotest_dir + 'Plots/Psat_fits/bolo23_Pfit' + fn_comments + '_forpaper.png' if save_figs else None
    # Gs, Ks, ns, Tcs, Gs_err, Ks_err, ns_err, Tcs_err = tes.fitPowerLaw(pRn, meas_temps, p_pnts.T, init_guess, fitToLast=True, 
    #                 suptitle='', TbsToReturn=TbsToReturn, plot=True, sigma=sigma_p.T, nstd=5, pfigpath=pfig_path)        
    # if save_figs: plt.savefig(bolotest_dir + 'Plots/for_paper/bolo23_fitparams' + fn_comments + '_forpaper.png', dpi=300) 
     



if estimate_Gcov:
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

        tes = tesanalyze_ahhh.TESAnalyze() 
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
        v_pnts = np.zeros((len(tlabels), len(pRn))); i_pnts = np.zeros((len(tlabels), len(pRn))); p_pnts = np.zeros((len(tlabels), len(pRn)))   # initialize interpolated IVs
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
            v_pnts[tt], i_pnts[tt], p_pnts[tt] = tes.ivInterpolate(vtes[tt], ites[tt], rtes[tt], pRn, rn_temp[tt], tran_pRn_start=tran_pRn_start, plot=False)


        TbsToReturn = meas_temps[tinds_return]
        if show_aplots: 
            plt.figure()
            tsort = np.argsort(meas_temps)
            for tt in tsort:
                plt.plot(v_pnts[tt]*1e6, i_pnts[tt]*1e3, 'o', alpha=0.7, color=plt.cm.plasma((meas_temps[tt]-min(TbsToReturn)*1)/(max(meas_temps))))   # meas_temps/max(meas_temps)
                plt.plot(vtes[tt]*1e6, ites[tt]*1e3, alpha=0.6, label='{} mK'.format(round(meas_temps[tt]*1E3,2)), color=plt.cm.plasma((meas_temps[tt]-min(TbsToReturn)*1)/(max(meas_temps)*0.8)))
            plt.xlabel('Voltage [$\mu$V]')
            plt.ylabel('Current [mA]')
            plt.title('Interpolated IV Points')
            plt.legend()
            if save_figs: plt.savefig(bolotest_dir + 'Plots/IVs/' + tesid + '_interpIVs' + fn_comments + '.png', dpi=300)

        rn = np.nanmean(rn_temp)
        rn_err = np.nanstd(rn_temp)

        Tb_toquote = meas_temps[Tb_ind]   # measured temperature to quote
        qind = np.where(pRn==pRn_toquote)[0][0]
        tind = np.where(TbsToReturn==Tb_toquote)[0][0]  # index of subset TbsToReturn to return
        sigma_p = np.zeros(np.shape(i_pnts))   # this is stupid but it works
        for ii, ipnt in enumerate(i_pnts):
            sigma_p[ii] = tes.sigma_power(i_pnts[ii], sigma_i[ii], v_pnts[ii], sigma_v[ii])

        # get initial fit for setting k and n ranges
        params_init = tes.fitPowerLaw(pRn, meas_temps, p_pnts.T, init_guess, fitToLast=True, 
                suptitle=tesid, TbsToReturn=TbsToReturn, plot=False, sigma=sigma_p.T, constT=True, fitGexplicit=False)     
        GTc0 = params_init[0][qind]; sigma_GTc0 = params_init[4][qind]
        K0 = params_init[1][qind]; sigma_K0 = params_init[5][qind]
        n0 = params_init[2][qind]; sigma_n0 = params_init[6][qind]

        params_fitG = tes.fitPowerLaw(pRn, meas_temps, p_pnts.T, init_guess, fitToLast=True, 
                suptitle=tesid, TbsToReturn=TbsToReturn, plot=False, sigma=sigma_p.T, constT=True, fitGexplicit=True)    
        GTc0_fitG = params_fitG[0][qind]; sigma_GTc0_fitG = params_fitG[4][qind]

        K_range = np.linspace(K0 - sigma_K0, K0 + sigma_K0)
        GTc_krange = np.zeros(len(K_range)); n_krange = np.zeros(len(K_range))
        init_guess_k = init_guess[[1,2]] 
        # pdb.set_trace()
        for kk, Kv in enumerate(K_range): 
            params_temp = tes.fitPowerLaw(pRn, meas_temps, p_pnts.T, init_guess_k, fitToLast=True, 
                suptitle=tesid, TbsToReturn=TbsToReturn, plot=False, sigma=sigma_p.T, nstd=5, constT=True, fitGexplicit=False, fixedK=Kv) 
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
            params_temp = tes.fitPowerLaw(pRn, meas_temps, p_pnts.T, init_guess_n, fitToLast=True, 
                suptitle=tesid, TbsToReturn=TbsToReturn, plot=False, sigma=sigma_p.T, nstd=5, constT=True, fitGexplicit=False, fixedn=nv) 
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

plt.show() 
