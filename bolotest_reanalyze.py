import numpy as np 
import matplotlib.pyplot as plt
import pickle as pkl
import tesanalyze_ahhh
import pdb
import csv 

# Reanalyze bolotest data from 2018 (?)
# AHHH 2020/12
#
# to do: add errors; save power law fit figures; fix SC branch of IVs; second y axis on Psat plot
# automated error analysis: baseline removal tes-by-tes basis scipy.signal.sabgal_filter


def G_atT(T, k, n):   # calculate thermal conductance at temperature T (in K)
    return n*k*T**(n-1)

def Psat_atT(T, Tc, k, n):
    return k*(Tc**n-T**n)

def sigma_power(i, sigma_i, v, sigma_v):   # calculate error in power measurement from error in voltage and current measurement
    # return np.sqrt(i**2*sigma_v**2 + v**2*sigma_i**2)   # sigma_power^2 = (I*sigma_V)^2 + (V*sigma_I)^2
    return np.sqrt(np.multiply(i**2,sigma_v**2) + np.multiply(v**2,sigma_i**2))   # sigma_power^2 = (I*sigma_V)^2 + (V*sigma_I)^2



### user params
analyze_ivs = False
save_data = False   # save csv and pkl
save_figs = False   

show_plots = True
plot_byind = False   # for easier IV exclusion 
plot_noisyivs = False
plot_mbolos = False
save_pad18 = False   # write quick CSV for Joel
scaleG = True

analysis_dir = '/Users/angi/NIS/Analysis/bolotest/'
dfiles = ['/Users/angi/NIS/MM2017L_20171117_data/AY_1thru15_IVvsTb.pkl', '/Users/angi/NIS/MM2017L_20171117_data/AX_16thru30_IVvsTb.pkl']
csv_file = analysis_dir + 'bolotest_reanalyzed_AHHH_202102_witherrors.csv'   # where to save analysis results
pkl_file = analysis_dir + 'bolotest_reanalyzed_AHHH_202102_witherrors.pkl'   # where to save analysis results

v_nfit = .3   # v_bias above which TES is normal (approximate)
pRn = np.array([25, 30, 40, 50, 60, 70, 80, 90])   # % Rn
tran_pRn_start = 0.2   # % Rn dubbed beginning of SC transition
# init_guess = [5.E-11, 3., 170.]   # kappa, n, Tc [mK]; powerlaw fitter
init_guess = [1.E-10, 2.5, .170]   # kappa, n, Tc [mK]; powerlaw fitter
v_offset = 0   # V; SC and/or normal branch should go through (0,0)
i_offset = 0.*1e-6   # Amps; SC and/or normal branch should go through (0,0)
tinds_return = np.array([0, 3, 8, 11, 15])
psat_ind = 0
pRn_toquote = 80   # % Rn fit values to print to screen
# sigma_v = 1e-10   # V, error in voltage measurement by eye from IV
# sigma_i = 1e-7   # A, error in current measurement
# sigma_v = 1e-3   # V, error in voltage measurement, testing
# sigma_i = 1e-3   # A, error in current measurement, testing

# readout circuit parameters
M_r = 8.5   # (SQUID-input coupling: current = arbs / (M_r * R_fb))
Rfb = 2070.   # Ohms (SQUID feeback resistor), Kelsey: accurate to ~5%
Rsh = 370.e-6   # microOhms (bias circuit shunt), Kelsey: accurate to 10-15%
Rb = 1020.   # Ohms (TES bias resistor), Kelsey: accurate to ~5%

if analyze_ivs:
    bays = ['BayAY', 'BayAX']
    pad_bolo_map = {'1':'1f', '2':'1e', '3':'1d', '4':'1c', '5':'1b', '6':'1a', '7':'1f', '8':'1e', '9':'1d', '10':'1c', '11':'1b', '12':'1a', 
                    '13':'24', '14':'23', '15':'22', '16':'13', '17':'21', '18':'20', '19':'24', '20':'23', '21':'22', '22':'13', '23':'21', 
                    '24':'20','25':'7','26':'26','27':'27','28':'7','29':'26','30':'27'}

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
    for ts, tesid in enumerate(tesids[0:1]):   # iterate through bolos

        bay = tesid.split('_')[0]; row = tesid.split('_')[1]
        tes = tesanalyze_ahhh.TESAnalyze() 
        ivs[tesid] = {}  
        tlabels = [key for key in data[bay][row]['iv']]
        if tesid == 'BayAY_Row12' or tesid == 'BayAX_Row10' or tesid == 'BayAX_Row03' or tesid == 'BayAX_Row00' or tesid == 'BayAX_Row06': 
            tlabels.remove('iv014')   # wonky IV
        maxiv = max([len(data[bay][row]['iv'][tlab]['data'][0]) for tlab in tlabels])   # handle IVs of different lengths
        asize = (len(data[bay][row]['iv']), maxiv)   # temp length by maximum iv size
        vbias = np.full(asize, np.nan); vfb = np.full(asize, np.nan)   # initialize arrays
        vtes = np.full(asize, np.nan); ites = np.full(asize, np.nan); rtes = np.full(asize, np.nan); i_meas = np.full(asize, np.nan); ptes = np.full(asize, np.nan)
        meas_temps = np.array([np.nan]*len(tlabels)); rn_temp = np.array([np.nan]*len(meas_temps))
        v_pnts = np.zeros((len(tlabels), len(pRn))); i_pnts = np.zeros((len(tlabels), len(pRn))); p_pnts = np.zeros((len(tlabels), len(pRn)))   # initialize interpolated IVs
        sigma_v = np.array([np.nan]*len(tlabels)); sigma_i = np.array([np.nan]*len(tlabels))

        plt.figure()
        for tt, tlab in enumerate(tlabels):   # iterate through temperatures

            meas_temps[tt] = data[bay][row]['iv'][tlab]['measured_temperature']
            ivlen = len(data[bay][row]['iv'][tlab]['data'][0])   # handle IVs of different lengths
            vbias[tt,:ivlen] = data[bay][row]['iv'][tlab]['data'][0,::-1]   # raw voltage, taken from high voltage -> 0
            vfb[tt,:ivlen] = data[bay][row]['iv'][tlab]['data'][1,::-1]   # raw current, taken from high voltage -> 0
            # vtes[tt], ites[tt], rtes[tt], ptes[tt], i_meas[tt], n_fit, sc_fit, rpar = tes.ivAnalyzeTDM(vbias[tt], vfb[tt], Rfb, Rb, Rsh, M_r, v_nfit, show_plot=False)   # *
            vtes[tt], ites[tt], rtes[tt], ptes[tt], i_meas[tt], n_fit, norm_inds, sc_fit, end_sc, rpar = tes.ivAnalyzeTDM(vbias[tt], vfb[tt], Rfb, Rb, Rsh, M_r, v_nfit, show_plot=False)   # *
            # norm_inds = np.where(vbias[tt] > v_nfit)
            # start_norm = min(norm_inds)
            # rn_temp[tt] = np.mean(rtes[tt, np.where(vbias[tt] > v_nfit)])   # ohms  * 
            # rn_temp[tt] = np.mean(rtes[tt, norm_inds])   # ohms
            rn_temp[tt] = np.mean(rtes[tt, norm_inds])   # ohms
            sigma_v[tt] = np.std(vtes[tt,:end_sc])   # V, error in voltage measurement by eye from IV
            sigma_i[tt] = np.std(ites[tt,norm_inds])  # A, error in current measurement

            # interpolate IV points    
            lab=tt if (plot_byind) else str(round(meas_temps[tt]*1e3)) + ' mK'
            plt.plot(vtes[tt]*1e6, ites[tt]*1e3, label=lab, alpha=0.7)
            plt.legend()
            v_pnts[tt], i_pnts[tt], p_pnts[tt] = tes.ivInterpolate(vtes[tt], ites[tt], rtes[tt], pRn, rn_temp[tt], tran_pRn_start=tran_pRn_start)

        plt.xlabel('Voltage [$\mu$V]')
        plt.ylabel('Current [mA]')
        plt.title('Interpolated IV Points')
        if save_figs: plt.savefig(analysis_dir + 'plots/' + tesid + '_interpIVs.png', dpi=300)

        if bay=='BayAY':
            ivs[tesid]['pad'] = str(int(tesid.split('Row')[1])+1)  # pad number
        elif bay=='BayAX':
            ivs[tesid]['pad'] = str(int(tesid.split('Row')[1])+16)  # pad number
        rn = np.nanmean(rn_temp)
        rn_err = np.nanstd(rn_temp)

        TbsToReturn = meas_temps[tinds_return]
        psat_Tb = meas_temps[psat_ind]
        qind = np.where(pRn==pRn_toquote)[0][0]
        tind = np.where(TbsToReturn==psat_Tb)[0][0]
        # sigma_p = sigma_power(i_pnts, sigma_i, v_pnts, sigma_v)
        sigma_p = np.zeros(np.shape(i_pnts))   # this is stupid but it works
        for ii, ipnt in enumerate(i_pnts):
            sigma_p[ii] = sigma_power(i_pnts[ii], sigma_i[ii], v_pnts[ii], sigma_v[ii])
        
        ### fit power law
        Gs, Ks, ns, Tcs, Gs_err, Ks_err, ns_err, Tcs_err = tes.fitPowerLaw(pRn, meas_temps, p_pnts.T, init_guess, fitToLast=True, 
                        suptitle=tesid, TbsToReturn=TbsToReturn, plot=True, sigma=sigma_p.T, nstd=5)
        if save_figs: plt.savefig(analysis_dir + 'plots/' + tesid + '_fitparams.png', dpi=300) 

        print(' ')
        print(' ')
        print(tesid)
        print('K = ', Ks[qind], ' +/- ', Ks_err[qind])
        print('n = ', round(ns[qind], 2), ' +/- ', round(ns_err[qind], 4))
        # print('Tc = ', round(Tcs[tind, qind]*1e3, 2), ' +/- ', Tcs_err[tind, qind]*1e3, 'mK')
        print('Tc = ', round(Tcs[qind]*1e3, 2), ' +/- ', Tcs_err[qind]*1e3, 'mK')
        print('G = ', round(Gs[tind, qind]*1e12, 2), ' +/- ', round(Gs_err[tind, qind]*1e12, 2), 'pW/K')
        print('TES Rn = ', round(rn*1e3, 2), ' +/- ', round(rn_err*1e3, 2), ' mOhms')

        ### calculate Psat
        # find transition + normal branch
        sc_inds = np.where((rtes[psat_ind]/rn)<.2)[0]
        start_ind = np.max(sc_inds)
        end_ind = np.max(np.where(((rtes[psat_ind]/rn)>.2) & (rtes[psat_ind]!=np.nan)))
        vtes_tran = vtes[psat_ind, start_ind:end_ind]
        ites_tran = ites[psat_ind, start_ind:end_ind]
        rtes_tran = rtes[psat_ind, start_ind:end_ind]

        # calculate Psat
        ptes_tran = vtes_tran * ites_tran
        sat_ind = np.where(ites_tran == np.min(ites_tran))[0][0]   # where the TES goes normal
        Psat = ptes_tran[sat_ind]
        Psat_err = sigma_power(ites_tran[sat_ind], sigma_i[psat_ind], vtes_tran[sat_ind], sigma_v[psat_ind])
        print('Psat = ', round(Psat*1e12, 4), ' +/- ', round(Psat_err*1e12, 4), 'pW')
        print('Psat (calc) = ', round(Psat_atT(.150, Tcs[qind], Ks[qind], ns[qind])*1e12, 4), 'pW')
        print(' ')
        print(' ')

        plt.figure()
        plt.plot(vtes_tran.T*1e6, ites_tran.T/np.max(ites_tran), label='TES IV')
        plt.plot(vtes_tran.T*1e6, ptes_tran.T/np.max(ptes_tran), label='Power')
        plt.plot(vtes_tran[sat_ind]*1e6, Psat/np.max(ptes_tran), 'x', label='$P_{sat}$')
        plt.xlabel('Voltage [$\mu$V]')
        plt.ylabel('Normalized Current')
        plt.legend()
        plt.title('TES IV and Calculated Power at Tbath = ' + str(round(psat_Tb*1000, 1)) + 'mK')
        if save_figs: plt.savefig(analysis_dir + 'plots/' + tesid + '_psatcalc.png', dpi=300)

        # store results in dict
        sort_inds = np.argsort(meas_temps)   # sort by temp, ignore nans
        ivs[tesid]['meas_temps'] = meas_temps[sort_inds]   # K
        ivs[tesid]['vbias'] = vbias[sort_inds]  
        ivs[tesid]['vfb'] = vfb[sort_inds]  
        ivs[tesid]['vtes'] = vtes[sort_inds]   # volts
        ivs[tesid]['ites'] = ites[sort_inds]   # amps
        ivs[tesid]['rtes'] = rtes[sort_inds]   # ohms
        ivs[tesid]['ptes'] = ptes[sort_inds]   # power
        ivs[tesid]['ptes_fit'] = p_pnts
        ivs[tesid]['ptes_err'] = sigma_p
        ivs[tesid]['i_meas'] = i_meas[sort_inds]   # amps
        ivs[tesid]['rn'] = rn   # ohms
        ivs[tesid]['rn_err'] = rn_err   # ohms
        ivs[tesid]['G@Tc [pW/K]'] = Gs[tind, qind]*1e12 
        ivs[tesid]['G@Tc_err [pW/K]'] = Gs_err[tind, qind]*1e12 
        ivs[tesid]['G@150 mK [pW/K]'] = G_atT(.150, Ks[qind], ns[qind])*1e12  
        ivs[tesid]['K'] = Ks[qind] 
        ivs[tesid]['K_err'] = Ks_err[qind] 
        ivs[tesid]['n'] = ns[qind] 
        ivs[tesid]['n_err'] = ns_err[qind] 
        ivs[tesid]['Tc [mK]'] = Tcs[qind]*1e3
        ivs[tesid]['Tc_err [mK]'] = Tcs_err[qind]*1e3
        ivs[tesid]['Psat@'+str(round(psat_Tb*1e3, 1))+'mK [pW]'] =  Psat*1e12
        ivs[tesid]['Psat_err@'+str(round(psat_Tb*1e3, 1))+'mK [pW]'] =  Psat_err*1e12
        ivs[tesid]['Psat@150mK [pW], Calc'] =  Psat_atT(.150, Tcs[qind], Ks[qind], ns[qind])*1e12
        # print(ns_err[qind])
        # print(ivs[tesid]['n_err'])
    
    if save_data:
        # write CSV     
        pads = [int(ivs[tesid]['pad']) for tesid in tesids]
        fields = ['Bolometer', 'Pad', 'Tc [mK]', 'Tc_err [mK]', 'Rn [mOhms]', 'Rn_err [mOhms]', 'G@Tc [pW/K]', 'G_err@Tc [pW/K]', 'G@150mK [pW/K]', 'k', 'k_err', 'n', 'n_err', 'Psat@'+str(round(psat_Tb*1e3, 1))+'mK [pW], IV', 'Psat@150mK [pW], Calc']  
        rows = [[pad_bolo_map[ivs[tesids[pp]]['pad']], ivs[tesids[pp]]['pad'], ivs[tesids[pp]]['Tc [mK]'], ivs[tesids[pp]]['Tc_err [mK]'], ivs[tesids[pp]]['rn']*1e3, ivs[tesids[pp]]['rn_err']*1e3, ivs[tesids[pp]]['G@Tc [pW/K]'], ivs[tesids[pp]]['G@Tc_err [pW/K]'], ivs[tesids[pp]]['G@150 mK [pW/K]'], 
                ivs[tesids[pp]]['K'], ivs[tesids[pp]]['K_err'], ivs[tesids[pp]]['n'], ivs[tesids[pp]]['n_err'], ivs[tesids[pp]]['Psat@'+str(round(psat_Tb*1e3, 1))+'mK [pW]'], ivs[tesids[pp]]['Psat_err@'+str(round(psat_Tb*1e3, 1))+'mK [pW]'], ivs[tesids[pp]]['Psat@150mK [pW], Calc']] for pp in np.argsort(pads)]
        # rows = [[pad_bolo_map[ivs[tesid]['pad']], ivs[tesid]['pad'], ivs[tesid]['Tc [mK]'], ivs[tesid]['Tc_err [mK]'], ivs[tesid]['rn']*1e3, ivs[tesid]['rn_err']*1e3, ivs[tesid]['G@Tc [pW/K]'], ivs[tesid]['G@Tc_err [pW/K]'], ivs[tesid]['G@150 mK [pW/K]'], 
                # ivs[tesid]['K'], ivs[tesid]['K_err'], ivs[tesid]['n'], ivs[tesid]['n_err'], ivs[tesid]['Psat@'+str(round(psat_Tb*1e3, 1))+'mK [pW]'], ivs[tesid]['Psat_err@'+str(round(psat_Tb*1e3, 1))+'mK [pW]'], ivs[tesid]['Psat@150mK [pW], Calc']] for tesid in np.argsort(tesids)]
        with open(csv_file, 'w') as csvfile:  
            csvwriter = csv.writer(csvfile)  # csv writer object  
            csvwriter.writerow(fields)  
            csvwriter.writerows(rows)

        # write pickle
        with open(pkl_file, 'wb') as pklfile:
            pkl.dump(ivs, pklfile)

if show_plots: plt.show() 

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
        plt.savefig(analysis_dir + 'plots/Noisy_IVs/' + tesid + '_rawIVs.png', dpi=300)

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
            vtes[tt], ites[tt], rtes[tt], ptes[tt], i_meas[tt], n_fit, sc_fit, rpar = tes.ivAnalyzeTDM(vbias[tt], vfb[tt], Rfb, Rb, Rsh, M_r, v_nfit, show_plot=False)
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
        plt.savefig(analysis_dir + 'plots/Missing_IVs/' + tesid + '_interpIVs.png', dpi=300)

        plt.figure()
        labels = [str(round(mtemp*1e3)) + ' mK' for mtemp in meas_temps]
        plt.plot(vbias.T, vfb.T)
        plt.legend(labels)
        plt.title(tesid)
        plt.xlabel('Vbias')
        plt.ylabel('Vfb')
        plt.savefig(analysis_dir + 'plots/Missing_IVs/' + tesid + '_rawIVs.png', dpi=300)

if save_pad18: # send P and T values for pad 18, bolo 20
    # with open(pkl_file, 'r') as pklfile:     
    #     ivs = pkl.load(pklfile)
    # pads = [int(ivs[tesid]['pad']) for tesid in tesids]
    tesid = [tesid for tesid in tesids if ivs[tesid]['pad']=='18'][0]
    fields = ivs[tesid]['meas_temps']
    # rows = [ivs[tesid]['ptes'], ivs[tesid]['meas_temps'], sigma_power(ivs[tesid]['ites'], sigma_i, ivs[tesid]['vtes'], sigma_v)]
    rows1 = ivs[tesid]['ptes_fit'].T
    rows2 = ivs[tesid]['ptes_err'].T
    with open('/Users/angi/NIS/Analysis/bolotest/pad18_pvt.csv', 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)  # csv writer object  
        csvwriter.writerow(fields)  
        csvwriter.writerows(rows1)
        csvwriter.writerows(rows2)

if scaleG:

    def scale_G(T, GTc, Tc, n):
        return GTc * T**(n-1)/Tc**(n-1)

    def sigma_GscaledT(T, GTc, Tc, n, sigma_GTc, sigma_Tc, sigma_n):
        Gterm = sigma_GTc * T**(n-1)/(Tc**(n-1))
        Tcterm = sigma_Tc * GTc * (1-n) * T**(n-1)/(Tc**(n-1))   # this is very tiny
        nterm = sigma_n * GTc * T**(n-1)/(Tc**(n-1)) * (np.log(T)-np.log(Tc))
        return np.sqrt( Gterm**2 + Tcterm**2 + nterm**2 )   # quadratic sum of sigma G(Tc), sigma Tc, and sigma_n terms

    Tcs = np.array([170.9877185, 170.9706609, 171.7552114, 171.7208656, 171.2176826, 170.9077816, 171.3470285, 171.0853375, 170.9200613, 171.060197, 170.0255488, 171.021025, 171.2850107, 170.4034122, 171.2042771, 170.2338939, 170.6389945, 172.0684101, 170.3518396, 171.721552, 171.5840882, 171.1586009, 170.8315709, 172.2051357, 171.7151257])
    sigma_Tcs = np.array([0.010421243, 0.009226958, 0.010790079, 0.009281995, 0.006323179, 0.009746988, 0.010994352, 0.009444727, 0.007650692, 0.005877343, 0.017378697, 0.009453003, 0.009177113, 0.021876877, 0.006768265, 0.006364612, 0.018369732, 0.012367842, 0.020356169, 0.007980188, 0.013312168, 0.006793048, 0.014202084, 0.009438768, 0.009855968])
    ns = np.array([2.102652493, 2.286305877, 2.181787081, 2.636751655, 3.625212281, 2.065019714, 2.030684323, 2.310936272, 2.271596966, 3.713230268, 2.05911454, 2.183641919, 2.389944922, 1.892766183, 3.29595625, 2.328520321, 2.105820885, 2.208529433, 1.994745782, 3.402832364, 2.135441834, 3.025155869, 2.821070825, 3.025157842, 2.816065888])
    sigma_ns = np.array([2.102652493, 2.286305877, 2.181787081, 2.636751655, 3.625212281, 2.065019714, 2.030684323, 2.310936272, 2.271596966, 3.713230268, 2.05911454, 2.183641919, 2.389944922, 1.892766183, 3.29595625, 2.328520321, 2.105820885, 2.208529433, 1.994745782, 3.402832364, 2.135441834, 3.025155869, 2.821070825, 3.025157842, 2.816065888])
    GTcs = np.array([7.063630682, 8.23553356, 10.31889778, 13.74098652, 21.98669484, 6.659975444, 7.601344581, 8.013653731, 10.63671194, 22.09412114, 4.67333846, 7.904637521, 10.03001622, 3.471988312, 16.47703326, 5.184876564, 5.196919583, 8.268319188, 3.703848367, 16.79390843, 5.659883891, 15.11216746, 10.07493674, 15.49026988, 10.44634148])
    sigma_GTcs = np.array([0.102018534, 0.112406438, 0.131803919, 0.171320173, 0.239205548, 0.092635138, 0.104237744, 0.1090491, 0.124941174, 0.238423651, 0.08774709, 0.108691126, 0.130040288, 0.072825133, 0.198879142, 0.078156549, 0.092319898, 0.113998559, 0.074520238, 0.204577595, 0.090894549, 0.167780174, 0.164340968, 0.18151003, 0.147841826])

    Tscale = .170
    scaledGs = scale_G(Tscale, GTcs, Tcs/1E3, ns)  # pW/K
    sigma_scaledGs = sigma_GscaledT(Tscale, GTcs, Tcs/1E3, ns, sigma_GTcs, sigma_Tcs/1E3, sigma_ns)

