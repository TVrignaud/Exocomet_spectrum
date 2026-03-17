import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from scipy.special import voigt_profile
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import convolve
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from bindensity import resampling    # Bourrier et al. 2025. Install with 'pip install bindensity'. See https://gitlab.unige.ch/delisle/bindensity
import copy
import plotly.graph_objects as go
import sys
import json

import List_studied_lines as List_studied_lines

plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'












'''

Constants and utils

'''

pi = np.pi


const_lambda_0 = 2756.5507        # A
const_h = 6.62607015e-34          # Js
const_hbar = const_h/2/pi      # Js
const_c = 299792458               # m/s
const_c_km = 299792.458           # km/s
const_eps_0 = 8.85418782*10**-12  # si
const_m_e = 9.109383701*10**-31   # kg
const_m_H = 1.672e-27             # kg
const_e = 1.602176634*10**-19     # C
const_k_b = 1.380649*10**-23      # si
const_SB = 5.670374e-8            # si
const_G = 6.67430e-11             # si
const_Msun = 1.9885e30            # m 
const_Rsun = 6.96342e8            # m 
const_Lsun = 3.826e26             # W 
const_pc = 3.085677581e16         # m
const_AU = 149597870000           # m  
const_cm_to_K = const_h*const_c/const_k_b*100    # Energy conversion between cm-1 and K (unit : K/cm-1)
const_eV_to_cm = const_e/(const_h*const_c)/100   # Energy conversion between eV and cm-1 (unit : cm-1/eV)
const_Ry_to_cm = 109737.31                       # Energy conversion between Ry and cm-1 (unit : cm-1/Ry)
const_Beta_col = (2*pi*const_hbar**4 / (const_k_b * const_m_e**3)) ** 0.5    # Beta factor used to calculate collision strength


T_eff_beta_Pic = 8000 # K
Lbeta_pic_sun = 8.7 # Lsun
M_beta_pic = 1.75  * 2 * 10**30    # kg
R_beta_pic = (Lbeta_pic_sun*const_Lsun/(4*pi*const_SB*T_eff_beta_Pic**4))**0.5    # 1.53 R_sun
AU_R_beta_pic = const_AU/R_beta_pic
rv_Beta_Pic = 20.5




def def_edge_tab(cen_bins):
    # Return the bin edges of a RV or wavelength table

    mid_bins = 0.5*(cen_bins[0:-1]+cen_bins[1::])
    low_bins_st =cen_bins[0] - (mid_bins[0] - cen_bins[0])
    high_bins_end = cen_bins[-1] + (cen_bins[-1]-mid_bins[-1])
    edge_bins =  np.concatenate(([low_bins_st], mid_bins,[high_bins_end]))

    return edge_bins


def vacuum_to_air(wl, Temperature = 15., Pressure = 760.) : 
    # Conversion from vaccum wavelength to air wavelength

    n = 1 + 1e-6 * Pressure * (1 + (1.049-0.0157*Temperature)*1e-6*Pressure) / 720.883 / (1 + 0.003661*Temperature) * (64.328 + 29498.1/(146-(1e4/wl)**2) + 255.4/(41-(1e4/wl)**2))
    return wl/n


def air_to_vacuum(wl, Temperature = 15., Pressure = 760.) : 
    # Conversion from air wavelength to vaccum wavelength

    wl_temp = wl
    for i in range(3) : 
        n = 1 + 1e-6 * Pressure * (1 + (1.049-0.0157*Temperature)*1e-6*Pressure) / 720.883 / (1 + 0.003661*Temperature) * (64.328 + 29498.1/(146-(1e4/wl_temp)**2) + 255.4/(41-(1e4/wl_temp)**2))
        wl_temp = wl*n

    return wl*n


def lorentzian_average(x,dx,delta) : 
    # Returns the integral of a lorentzian profile of width delta from x-dx/2 to x+dx/2
    a = x - dx/2
    b = x + dx/2
    return (1.0 / pi) * (np.arctan(b/delta) - np.arctan(a/delta))


def give_mass(sp):
    # Mass of various atoms and ions
    dic_mass = {
        'H I': 1.008, 'H II': 1.008,
        'C I': 12.011, 'C II': 12.011, '13C I': 13.003, '13C II': 13.003,
        'C I 3P 1/2' : 12.011, 'C I 3P 3/2' : 12.011, 'C I 3P 5/2' : 12.011, 'C I 1D 5/2' : 12.011, 'C I 1S 1/2' : 12.011, 
        '13C I 3P 1/2' : 13.003, '13C I 3P 3/2' : 13.003, '13C I 3P 5/2' : 13.003, '13C I 1D 5/2' : 13.003, '13C I 1S 1/2' : 13.003, 
        'N I': 14.007, 'N II': 14.007,
        'O I': 15.999, 'O II': 15.999,
        'Na I': 22.990, 'Na II': 22.990,
        'Mg I': 24.305, 'Mg II': 24.305,
        'Al I': 26.982, 'Al II': 26.982,
        'Si I': 28.085, 'Si II': 28.085,
        'S I': 32.06, 'S II': 32.06,
        'Ca I': 40.078, 'Ca II': 40.078,
        'Cr I': 51.996, 'Cr II': 51.996,
        'Mn I': 54.938, 'Mn II': 54.938,
        'Fe I': 55.845, 'Fe II': 55.845,
        'Co I': 58.933, 'Co II': 58.933,
        'Ni I': 58.693, 'Ni II': 58.693,
        'Zn I': 65.38, 'Zn II': 65.38,
    }
    return dic_mass[sp]



'''

Pipeline which builds the "Analysis_dic" used to fit spectroscopic observations of exocomets, using various options specified in 'settings'

+ Retrieves spectroscopic observations
+ Retrieves tabulated data for the studied species (Fe II, Ni II, C I ...): energy levels, transition wavelength, oscillator strengths, collision rates
+ Retrieves other things required to reproduce the data : Line Spread Functions (LSF), signatures for the Interstellar Medium (ISM), Hyperfine Structures (HFS) for specific lines...

Once build, 'Analysis_dic' allows to quickly compute the transmission model for any set of exocomet parameters (d: distances to the star; N: column densities; v: radial velocities), 
and compare the model to data (resample at instrumental resolution; compute the Chi2, etc.)


'''

def Prepare_analysis(settings) : 

    Analysis_dic = {}

    # Retrieve observed spectra 
    Retrieve_observed_spectra(Analysis_dic, settings)

    # Retrieve tabulated data for required species 
    Retrieve_species_prop(Analysis_dic, settings)

    # Build HR wavelength tables
    Build_HR_wl_table(Analysis_dic, settings)

    # Retrieve line parameters, from NIST + tabulated data
    Retrieve_lines(Analysis_dic, settings)

    # Retrieve the flux of Beta Pic in the studied lines
    Retrieve_line_fluxes(Analysis_dic, settings)

    # Retrieve ISM properties
    Retrieve_ISM_prop(Analysis_dic, settings)

    # Retrieve LSF of STIS
    Retrieve_LSF(Analysis_dic, settings)

    # Identify which lines and which orders will be ultimately fitted
    Identify_fitted_orders(Analysis_dic, settings)

    # Retrieve lines with hyperfine structure
    Retrieve_HFS(Analysis_dic, settings)

    return Analysis_dic






# Retrieves spectroscopic data
#    + The spectra have already been reduced and are located in Data_Beta_Pic.py
#    + Each spectrum is divided by a continuum (specified in settings) to extract the exocomets absorption spectrum
#    + The spectra can be truncated to analysis only a fraction of it
#    + The level of zero-flux can beincreased if necessary 
def Retrieve_observed_spectra(Analysis_dic, settings) : 

    Analysis_dic['Observed_spectra'] = {}
    Analysis_dic['Continuum'] = {}

    spec_dic = np.load(settings['path_data'], allow_pickle = True).item()

    # Iterate over instruments/dates/spectra/orders
    for inst in settings['list_spectra'] : 
        func_continuum = spec_dic['reference'][inst][ settings['continuum'][inst] ]['spline']
        Analysis_dic['Continuum'][inst] = {'spline' : func_continuum, 
                                           'valid_domains' : spec_dic['reference'][inst][ settings['continuum'][inst] ]['valid_domains']}
        Analysis_dic['Observed_spectra'][inst] = {}
        for date in settings['list_spectra'][inst] :
            Analysis_dic['Observed_spectra'][inst][date] = {}
            for spec in settings['list_spectra'][inst][date] : 
                Analysis_dic['Observed_spectra'][inst][date][spec] = {}
                for i_ord in spec_dic['spec_renorm'][inst][date][spec] :
                    
                    # Extract spectra
                    wl_spec  = copy.deepcopy(spec_dic['spec_renorm'][inst][date][spec][i_ord]['wl'])
                    flux_spec = copy.deepcopy(spec_dic['spec_renorm'][inst][date][spec][i_ord]['flux'])
                    error_spec = copy.deepcopy(spec_dic['spec_renorm'][inst][date][spec][i_ord]['error'])

                    # Normalise by continuum to extract the exocomets spectrum
                    continuum_loc = func_continuum(wl_spec)
                    flux_spec /= continuum_loc
                    error_spec /= continuum_loc

                    # Bring wl tables in Beta Pic rest frame
                    wl_spec /= (1 + rv_Beta_Pic/const_c_km)

                    # Keep only the studied part of the spectrum
                    if inst in settings['truncate_data'] : 
                        cond_kept = (wl_spec > settings['truncate_data'][inst][0]) & (wl_spec < settings['truncate_data'][inst][1])
                        wl_spec    = wl_spec   [cond_kept]
                        flux_spec  = flux_spec [cond_kept]
                        error_spec = error_spec[cond_kept]

                    # Increase the level of zero-flux if required
                    if inst in settings['increase_zero_flux_level'] : 
                        for x in settings['increase_zero_flux_level'][inst] :
                            plage = x[0]
                            zero_flux = x[1]
                            cond_correction = (wl_spec > plage[0]) & (wl_spec < plage[1])
                            continuum = func_continuum(wl_spec[cond_correction]*(1 + rv_Beta_Pic/const_c_km))
                            flux_spec[cond_correction] = flux_spec[cond_correction]*(1 + zero_flux/continuum) - zero_flux/continuum

                    # Store the observed exocomet spectrum            
                    Analysis_dic['Observed_spectra'][inst][date][spec][i_ord] = {'wl' : wl_spec, 'flux_renorm' : flux_spec, 'error_renorm' : error_spec}

    
    return None


# Function to interpolate a list of (T_e, upsilon), where T_e is the electronic temperature and upsilon the effective collision strength
def make_function_upsilon(list_log_T_e, upsilon) : 
    fit_spline = CubicSpline(list_log_T_e, np.log(upsilon + 1e-5), bc_type = 'natural')
    def fonc_upsilon(log_T_e) :
        if log_T_e < 2 : return np.exp(fit_spline(2))
        if log_T_e > 5 : return np.exp(fit_spline(5))
        return np.exp(fit_spline(log_T_e))
    
    return fonc_upsilon


# Retrieves datasets of the spectroscopic properties of the studied species
#    + Dataset have been pre-computed in data_fe2.npy, data_c1.npy, etc. 
#    + For each species, they contain: 
#        - The list of all energy levels 
#        - The table of all transitions between these levels
#        - The oscillator strengths and effective collision strength of all the transition (when available)
#    + These datasets are not always accurate (e.g. wavelenths, transition probabilities), but are very complete. Their role is primarily to compute the excitation state of the studied species. 
def Retrieve_species_prop(Analysis_dic, settings) : 

    Analysis_dic['List_species']        = settings['list_species']

    Analysis_dic['Data_species'] = {}
    for sp in Analysis_dic['List_species'] : 

        # Case where the excitation state of sp is modelled (e.g. Fe II)
        if settings['n_level_modeled'][sp] >= 2 : 

            data = np.load(settings['path_species'][sp], allow_pickle=True).item()

            Analysis_dic['Data_species'][sp] = data
            Analysis_dic['Data_species'][sp]['n_level_max'] = len(data['E_cm'])
            Analysis_dic['Data_species'][sp]['n_level_modeled'] = settings['n_level_modeled'][sp]


            Analysis_dic['Data_species'][sp]['spline_upsilon'] = {}
            list_T_e = data['e_temp']
            list_log_T_e = np.log10(list_T_e)

            # Build cublic splines to calculate the effective collision strength (upsilon) from any electronic temperature
            for i in range(settings['n_level_modeled'][sp]) : 
                for j in range(settings['n_level_modeled'][sp]) : 
                    upsilon = data['upsilon'][i,j]
                    if np.any(upsilon > 0) : Analysis_dic['Data_species'][sp]['spline_upsilon'][(i,j)] = make_function_upsilon(list_log_T_e, upsilon) 
                    else : Analysis_dic['Data_species'][sp]['spline_upsilon'][(i,j)] = None

        # Case where only one level is studied (e.g. C I 3P 1/2)
        else : 
            Analysis_dic['Data_species'][sp] = {'n_level_max' : 1, 'n_level_modeled' : 1}


    

    return None


# Build a high-resolution wavelength table used to compute the full transmission spectrum of the exocomets
def Build_HR_wl_table(Analysis_dic, settings) : 

    fact = 1 + (settings['res_rv_HR']/const_c*1e3)   # Factor from one wavelength pixel to the next
    wl_min, wl_max = settings['wl_range_global']
    wl_temp = wl_min

    table_wl_HR = []
    while wl_temp <= wl_max : 
        table_wl_HR.append(wl_temp)
        wl_temp*=fact

    Analysis_dic['table_wl_HR'] = np.array(table_wl_HR)
    
    return None


#  Build dictionnaries for the studied spectral lines
#    + 'Lines_fitted' : list of the lines which will utimately be fitted. 
#         - For these lines, precise oscillator strengths and wavelengths are provided in List_studied_lines.py. 
#         - The line parameters are mostly extracted from NIST
#    + 'Lines_all' : all the lines from the studied species, included optically allowed UV/visible transiting, are forbidden IR transition between levels of same parity
#         - The NIST parameters are used when available (i.e. when provided in List_studied_lines.py), otherwise we use the full datasets
#         - Goal: compute the excitation state of the transiting gas
#    + 'Lines_saturated': list of the strongest lines, subject to saturation. 
#         - For these lines, we also model the probability that photons emitted by the gas through spntaneous emission are re-absorbed by the gas itself.
#         - The list of energy levels involved in at least one transition of 'Lines_saturated' is stored in 'Levels_saturated'
def Retrieve_lines(Analysis_dic, settings) :

    Analysis_dic['Lines'] = {'Lines_fitted' : {},             # Fitted lines
                             'Lines_all' : {},                # All lines of the studied species
                             'Lines_saturated' : {},          # Strongest lines for which the escape probability if computed
                             'Levels_saturated' : {},         # Levels involved in the strongest lines 
    }

    for sp in Analysis_dic['List_species'] : 

        
        # Case where the excitation states is modelled
        if Analysis_dic['Data_species'][sp]['n_level_modeled'] >= 2 : 

            n_level_max = Analysis_dic['Data_species'][sp]['n_level_max']
            n_level_modeled = Analysis_dic['Data_species'][sp]['n_level_modeled']
            tab_E_cm = Analysis_dic['Data_species'][sp]['E_cm']
            tab_mult = Analysis_dic['Data_species'][sp]['mult']
            tab_lambda = Analysis_dic['Data_species'][sp]['wl']
            tab_A = Analysis_dic['Data_species'][sp]['A']
            tab_f = Analysis_dic['Data_species'][sp]['f']  
            
            # Retrieve the properties of the fitted lines
            #    + For each line of List_studied_lines.py (characterised by a wavelength and a lower energy level), we find the corresponding energy indexes (i,j) from the tabulated datasets.
            dic_ij_specific = {'Ca II' : {1838.01    : (1, 10),    1840.06   : (2, 11)},
                               'Cr II' : {2066.1638  : (0, 27),    2669.5023 : (2, 33),    2673.6222 : (4, 36),   2677.955  : (4, 35),
                                          2677.9538  : (5, 37),    2767.3539 : (5, 29),    2836.4656 : (5, 26),   2868.4894 : (1, 21),
                                          2876.8337  : (9, 45),    2699.4837 : (1, 31),    2861.7742 : (1, 22)   },   
                                'S I'   : {1270.7864 : (0, 29),    1270.7797 : (0, 30),    1316.5423 : (0, 23)   },
                                'Fe II' : {1788.0039 : (35, 213),  1788.0827 : (35, 212),  2423.4238 : (50, 178) }
            }


            dic_lines_fitted_sp = {}   

            for x in List_studied_lines.dic_all_lines[sp]: 
                i_line, j_line = None, None
                wl, f, E_cm, g_l, g_u = x[0], x[1], x[2], x[3], x[4]
                count = 0

                # Recover i,j directly from the species and wavelength 
                if sp in dic_ij_specific and wl in dic_ij_specific[sp] :
                    i_line, j_line = dic_ij_specific[sp][wl]
                    count = 1

                # Otherwise, search the tabulated data to find the couple (i,j) associated to the line
                else : 
                    for i in range(min(60, n_level_max)) : 
                        if (sp == 'Cr II' and (np.abs(tab_E_cm[i] - E_cm) < 50)) or ((sp != 'Cr II') and (np.abs(tab_E_cm[i] - E_cm) < 5)) : 
                            for j in range(i+1,n_level_max) :
                                wl_ij_obs = vacuum_to_air(tab_lambda[i,j]) if tab_lambda[i,j] >= 3250 else tab_lambda[i,j]
                                if (sp == 'Cr II' and (np.abs(wl_ij_obs - wl) < 5)) or (sp == 'C I' and (np.abs(wl_ij_obs - wl) < 0.01)) or ((sp not in ['C I', 'Cr II']) and (np.abs(wl_ij_obs - wl) < 0.1)) :  
                                    i_line, j_line = i,j
                                    count += 1

                # Check if line has been found once and only once
                if count == 0   : print(f'Fitted line from {sp} at {wl} not found in tabulated data')     
                elif count >= 2 : print(f'Fitted line from {sp} at {wl} found multiple times in tabulated data')  

                # Store data
                if i_line < Analysis_dic['Data_species'][sp]['n_level_modeled'] : 
                    A = f / ( const_eps_0*const_m_e*const_c**3/(2*pi*const_e**2*(const_c/wl*1e10)**2) * g_u/g_l   )
                    b_nat_wl = (wl*1e-10)**2/const_c * A/(4*pi)*1e10
                    dic_lines_fitted_sp[(i_line,j_line)] = {'wl' : wl, 'A' : A, 'f' : f, 'gf' : g_l*f,   'b_nat_wl' : b_nat_wl, 'E_low_cm' : E_cm}


                # Retrieve the properties of all the lines of the current species
                #    + 'dic_lines_all_sp' includes all optically allowed UV/optical transitions and parity-forbidden IR lines
                #    + 'dic_lines_saturated_sp' stores lines which are susceptible to be saturated
                #    + To estimate which lines are likely to be saturated, we use the criterion gf*wl/const_lambda_0*np.exp(-E_l/7000) > criterion_saturated[sp]
                dic_lines_all_sp     = {}      
                dic_lines_saturated_sp  = {}      
                list_levels_saturated_sp = []                
                criterion_saturated = {'Fe II' : 0.015, 'Ni II' : 0.06, 'Ca II' : 0.015, 'Cr II' : 0.27, 'Mn II' : 0.32, 'Si II' : 0.0014, 'Co II' : 3, 'S I' : 0.01, 'S II' : 0.001, 'Sr II' : 10}
                for i in range(n_level_modeled) : 
                    for j in range(i+1,n_level_modeled) : 
                        if  tab_A[i,j] > 0. :
                            if (i,j) in dic_lines_fitted_sp : dic_lines_all_sp[(i,j)] = dic_lines_fitted_sp[(i,j)]
                            else                            : dic_lines_all_sp[(i,j)] = {'wl' : tab_lambda[i,j], 'A' : tab_A[i,j], 'f' : tab_f[i,j], 'gf' : tab_mult[i]*tab_f[i,j],  'b_nat_wl' : (wl*1e-10)**2/const_c * tab_A[i,j]/(4*pi)*1e10,  'E_low_cm' : tab_E_cm[i]}

                            if (tab_mult[i]*tab_f[i,j]*tab_lambda[i,j]/const_lambda_0*np.exp(-tab_E_cm[i]*const_cm_to_K/7000) > criterion_saturated[sp]) or (sp in ['Fe II', 'Si II', 'C I', 'Ni II', 'Mn II'] and (tab_E_cm[i] < 10 and tab_A[i,j] > 1e5)) : 
                                dic_lines_saturated_sp[(i,j)] = dic_lines_all_sp[(i,j)]
                                list_levels_saturated_sp.append(i)
                                list_levels_saturated_sp.append(j)

                list_levels_saturated_sp = np.unique(list_levels_saturated_sp)



        # Case where only one level studied
        if Analysis_dic['Data_species'][sp]['n_level_modeled'] == 1 : 
        
            dic_lines_fitted_sp = {}  
            i_line, j_line = 0, 1
            for x in List_studied_lines.dic_all_lines[sp]: 
                wl, f, E_cm, g_l, g_u = x[0], x[1], x[2], x[3], x[4]
                A = f / ( const_eps_0*const_m_e*const_c**3/(2*pi*const_e**2*(const_c/wl*1e10)**2) * g_u/g_l   )
                b_nat_wl = (wl*1e-10)**2/const_c * A/(4*pi)*1e10
                dic_lines_fitted_sp[(0,j_line)] = {'wl' : wl, 'A' : None, 'f' : f, 'gf' : g_l*f,   'b_nat_wl' : b_nat_wl, 'E_low_cm' : 0}
                j_line += 1

            dic_lines_all_sp     = {}      
            dic_lines_saturated_sp  = {}      
            list_levels_saturated_sp = []   


        # For fitted lines and saturated lines, we also retrieve the pixels from Analysis_dic['table_wl_HR'] which are close to the line, to accelerate the computation. 
        for (i,j) in dic_lines_fitted_sp :
            wl = dic_lines_fitted_sp[(i,j)]['wl']
            table_wl_HR = Analysis_dic['table_wl_HR']
            rv_HR = (table_wl_HR - wl)/wl * const_c_km
            cond_close_line_HR = (rv_HR > settings['RV_window_model'][0]) & (rv_HR < settings['RV_window_model'][1])
            dic_lines_fitted_sp[(i,j)]['ind_close_HR'] = np.where(cond_close_line_HR)
            dic_lines_fitted_sp[(i,j)]['wl_close_HR'] = table_wl_HR[cond_close_line_HR]
            dic_lines_fitted_sp[(i,j)]['rv_close_HR'] = rv_HR[cond_close_line_HR]
            dic_lines_fitted_sp[(i,j)]['rv_close_HR_edge'] = def_edge_tab(rv_HR[cond_close_line_HR])

        for (i,j) in dic_lines_saturated_sp :
            wl = dic_lines_saturated_sp[(i,j)]['wl']
            table_wl_HR = Analysis_dic['table_wl_HR']
            rv_HR = (table_wl_HR - wl)/wl * const_c_km
            cond_close_line_HR = (rv_HR > settings['RV_window_model'][0]) & (rv_HR < settings['RV_window_model'][1])
            dic_lines_saturated_sp[(i,j)]['ind_close_HR'] = np.where(cond_close_line_HR)
            dic_lines_saturated_sp[(i,j)]['wl_close_HR'] = table_wl_HR[cond_close_line_HR]
            dic_lines_saturated_sp[(i,j)]['rv_close_HR'] = rv_HR[cond_close_line_HR]
            dic_lines_saturated_sp[(i,j)]['rv_close_HR_edge'] = def_edge_tab(rv_HR[cond_close_line_HR])

        # Store line data
        Analysis_dic['Lines']['Lines_fitted'][sp]     = dic_lines_fitted_sp
        Analysis_dic['Lines']['Lines_saturated'][sp]  = dic_lines_saturated_sp
        Analysis_dic['Lines']['Lines_all'][sp]        = dic_lines_all_sp
        Analysis_dic['Lines']['Levels_saturated'][sp] = list_levels_saturated_sp
        
    return None


# Retrieve the unnoculted flux of Beta Pic in the studied lines
#    + This is important to compute the excitation of the studied species in the radiation-dominated regime
#    + The spectrum is stored in Data_tabulated/Model_spectrum_Beta_Pic.npy
def Retrieve_line_fluxes(Analysis_dic, settings) :

    Analysis_dic['Reference_spectrum'] = {}

    dic_spec = np.load(settings['path_reference_spectrum'], allow_pickle = True).item()

    Analysis_dic['Reference_spectrum']['nu']   = dic_spec['nu']
    Analysis_dic['Reference_spectrum']['F_nu'] = dic_spec['F_nu']    
    Analysis_dic['Reference_spectrum']['lambda']   = dic_spec['lambda']
    Analysis_dic['Reference_spectrum']['F_lambda'] = dic_spec['F_lambda']

    Analysis_dic['Reference_spectrum']['Specific_intensity_lines'] = {}
    dic_spec = np.load(settings['path_reference_spectrum'], allow_pickle = True).item()
    tab_nu_spec = dic_spec['nu'] 

    for sp in Analysis_dic['Lines']['Lines_all'] : 
        if Analysis_dic['Data_species'][sp]['n_level_modeled'] >= 2 : 
            n_level_max = Analysis_dic['Data_species'][sp]['n_level_max']
            Specific_intensity_lines = np.zeros((n_level_max,n_level_max))
            for (i,j) in Analysis_dic['Lines']['Lines_all'][sp] :
                wl_line = Analysis_dic['Lines']['Lines_all'][sp][(i,j)]['wl']
                nu_line = const_c / (wl_line*1e-10)
                
                ind = np.searchsorted(tab_nu_spec, nu_line) - 1

                Specific_intensity_lines[i,j] = dic_spec['F_nu'][ind]
                Specific_intensity_lines[j,i] = dic_spec['F_nu'][ind]

            Analysis_dic['Reference_spectrum']['Specific_intensity_lines'][sp] = Specific_intensity_lines

    return None


# Retrieve properties of the ISM towards Beta Pic
#    + Column densities, line widths, radial velocities
def Retrieve_ISM_prop(Analyis_dic, settings) : 
    Analyis_dic['ISM'] = settings['ISM']


# Retrieve the line spread functions of the instruments used:
#    + For STIS and COS, they are retrieved from the handbook and provided in Data_tabulated/LSF
#    + For HARPS, we use Brandeker+ 2011.
#    + When the LSF has a constant width in radial velocity (HARPS + STIS), the kernel is stored at the same resolution of Analysis_dic['table_wl_HR'] (to facilitate the convolution)
#    + When the LSF has a constant width in wavelength (COS), the kernel is stored at a resolution of 0.001 A
def Retrieve_LSF(Analysis_dic, settings) : 

    configs_STIS = {

    'E230H_2400_0_01x0_03' : ['E230H_2400.txt', '0.1x0.03', 1.27],   # file, slit, rv size of one pixel
    'E230H_2400_0_1x0_2'   : ['E230H_2400.txt', '0.1x0.2', 1.27],
    'E230H_2400_0_1x0_09'  : ['E230H_2400.txt', '0.1x0.09', 1.27],
    'E230H_2400_6x0_2'     : ['E230H_2400.txt', '6x0.2', 1.27],
    'E230H_1700_0_01x0_03' : ['E230H_1700.txt', '0.1x0.03', 1.27],
    'E230H_1700_0_1x0_2'   : ['E230H_1700.txt', '0.1x0.2', 1.27],
    'E230H_1700_0_1x0_09'  : ['E230H_1700.txt', '0.1x0.09', 1.27],
    'E230H_1700_6x0_2'     : ['E230H_1700.txt', '6x0.2', 1.27],

    'E140H_1500_0_01x0_03' : ['E140H_1500.txt', '0.1x0.03', 1.32],
    'E140H_1500_0_2x0_09'  : ['E140H_1500.txt', '0.2x0.09', 1.32],
    'E140H_1500_0_2x0_2'   : ['E140H_1500.txt', '0.2x0.2', 1.32],
    'E140H_1500_6x0_2'     : ['E140H_1500.txt', '6x0.2', 1.32],
    'E140H_1200_0_01x0_03' : ['E140H_1200.txt', '0.1x0.03', 1.32],
    'E140H_1200_0_2x0_09'  : ['E140H_1200.txt', '0.2x0.09', 1.32],
    'E140H_1200_0_2x0_2'   : ['E140H_1200.txt', '0.2x0.2', 1.32],
    'E140H_1200_6x0_2'     : ['E140H_1200.txt', '6x0.2', 1.32],

    }

    Analysis_dic['LSF'] = {}

    # Iterate on inst/date/spec
    for inst in  Analysis_dic['Observed_spectra'] : 
        Analysis_dic['LSF'][inst] = {}
        for date in Analysis_dic['Observed_spectra'][inst] : 
            Analysis_dic['LSF'][inst][date] = {}
            for spec in Analysis_dic['Observed_spectra'][inst][date] : 

                config = settings['LSF'][inst][date][spec]


                if inst == 'STIS' :

                    grating = configs_STIS[config][0]
                    aperture = configs_STIS[config][1]
                    conversion_pixel_rv = configs_STIS[config][2]

                    file = pd.read_csv(settings['LSF_path']+'//'+grating, sep=r'\s+', skiprows=1)
                    pixels = np.array(file['Rel_pixel'])
                    lsf = np.array(file[aperture])

                    step_HR = settings['res_rv_HR']
                    n_pixels_HR = 100
                    rv_HR = np.linspace(-n_pixels_HR*step_HR, n_pixels_HR*step_HR, 2*n_pixels_HR + 1)
                    pixels_HR = rv_HR/conversion_pixel_rv

                    kernel = resampling(def_edge_tab(pixels_HR), def_edge_tab(pixels), lsf, kind = 'cubic')
                    kernel[np.isnan(kernel)] = 0
                    kernel /= np.sum(kernel)

                    Analysis_dic['LSF'][inst][date][spec] = {'kernel' : kernel, 'mode' : 'rv'}


                if inst == 'COS' :

                    file_fp3 = pd.read_csv(settings['LSF_path']+'//'+'G130M_1291_FP3.dat', sep=r'\s+', skiprows=0)
                    file_fp4 = pd.read_csv(settings['LSF_path']+'//'+'G130M_1291_FP4.dat', sep=r'\s+', skiprows=0)

                    pixels = np.arange(-160,161)
                    n_pixels = len(pixels)

                    lsf_averaged = np.zeros(n_pixels)

                    for key in file_fp3 : 
                        lsf_averaged += file_fp3[key]

                    for key in file_fp4 : 
                        lsf_averaged += file_fp4[key]

                    fit_spline_lsf = CubicSpline(pixels, lsf_averaged, bc_type = 'natural')

                    step_wl_HR = 0.001
                    step_pixel_HR = step_wl_HR/0.0099688017564
                    n_pixels_HR = 500

                    pixels_HR = np.linspace(-n_pixels_HR*step_pixel_HR, n_pixels_HR*step_pixel_HR, 2*n_pixels_HR + 1)
                    kernel = fit_spline_lsf(pixels_HR)
                    kernel /= np.sum(kernel)

                    Analysis_dic['LSF'][inst][date][spec] = {'kernel' : kernel, 'mode' : 'wl', 'step' : step_wl_HR}


                if inst == 'HARPS' : 

                    step_HR = settings['res_rv_HR']
                    n_pixels_HR = 100
                    rv_HR = np.linspace(-n_pixels_HR*step_HR, n_pixels_HR*step_HR, 2*n_pixels_HR + 1)

                    # Brandeker 2011
                    kernel = 1.193 * np.exp(-0.5 * (rv_HR - 0.00564)**2/1.135**2) - 0.194 * np.exp(-0.5 * (rv_HR + 0.00901)**2/0.680**2)
                    kernel /= np.sum(kernel)

                    if 'adjust_LSF_HARPS' in settings : 
                        b_gauss_wings = settings['adjust_LSF_HARPS']['b_gauss_wings']
                        weight_wings = settings['adjust_LSF_HARPS']['weight_wings']
                        kernel_wings = np.exp(-rv_HR**2/b_gauss_wings**2)
                        kernel_wings *= weight_wings/np.sum(kernel_wings)
                        kernel += kernel_wings
                        kernel /= np.sum(kernel)

                    Analysis_dic['LSF'][inst][date][spec] = {'kernel' : kernel, 'mode' : 'rv'}

    return None


# We identify the list of lines covered by each spectral order
#    + For STIS, there are 20-60 orders per spectrum
#    + For COS, this corresponds to the two spectral elements
#    + For HARPS, we simply use one 1D resampled spectrum. We thus treat it as if it was a spectrum with a unique order. 
def Identify_fitted_orders(Analysis_dic, settings) : 

    Analysis_dic['Fitted_orders'] = {}
    dof = 0
    fitted_lines = 0

    # Iterate on inst/data/spec/order
    for inst in Analysis_dic['Observed_spectra'] : 
        Analysis_dic['Fitted_orders'][inst] = {}
        rv_min_fit, rv_max_fit = settings['fitted_rv_range'][inst][0], settings['fitted_rv_range'][inst][1]
        
        for date in Analysis_dic['Observed_spectra'][inst] : 
            Analysis_dic['Fitted_orders'][inst][date] = {}
            for spec in Analysis_dic['Observed_spectra'][inst][date] : 
                Analysis_dic['Fitted_orders'][inst][date][spec] = {}

                for i_ord in Analysis_dic['Observed_spectra'][inst][date][spec] : 

                    wl_ord = Analysis_dic['Observed_spectra'][inst][date][spec][i_ord]['wl']
                    flux_renorm = Analysis_dic['Observed_spectra'][inst][date][spec][i_ord]['flux_renorm']
                    error_renorm = Analysis_dic['Observed_spectra'][inst][date][spec][i_ord]['error_renorm']


                    # Shift the wavelength table if required
                    #    + The wl calibration of STIS and COS is often found to be imperfect, by ~ 0-1 km/s for STIS, 0-10 km/s for COS
                    if inst in settings['RV_shift_lines'] and date in settings['RV_shift_lines'][inst] and spec in settings['RV_shift_lines'][inst][date] and i_ord in settings['RV_shift_lines'][inst][date][spec] : 
                        for x in settings['RV_shift_lines'][inst][date][spec][i_ord] :
                            plage_wl = x[0]
                            shift_rv = x[1]
                            cond_shift = (wl_ord > plage_wl[0]) & (wl_ord < plage_wl[1])
                            wl_ord[cond_shift] *= (1 - shift_rv/const_c_km) 

                            cond_keep = np.ones_like(wl_ord, dtype = bool)
                            while np.any(np.diff(wl_ord[cond_keep]) < 0) : 
                                cond_remove_step = np.diff(wl_ord[cond_keep]) < 0
                                ind_keep_step = np.where(cond_remove_step)[0]
                                cond_keep[ind_keep_step] = False

                            wl_ord       = wl_ord[cond_keep]
                            flux_renorm  = flux_renorm[cond_keep]
                            error_renorm = error_renorm[cond_keep]

                    # Check the list of fitted lines to see which are covered by the current order
                    cond_fit_order = np.zeros(len(wl_ord), dtype = bool)
                    dic_lines_order = {}

                    for sp in Analysis_dic['List_species'] : 
                        dic_lines_order[sp] = {}

                        for (i,j) in Analysis_dic['Lines']['Lines_fitted'][sp] : 
                            wl = Analysis_dic['Lines']['Lines_fitted'][sp][(i,j)]['wl']

                            # Check if at least one pixel of the order is 'close' to the line
                            cond_close_line = (wl_ord > wl*(1 + rv_min_fit/const_c_km)) & (wl_ord < wl*(1 + rv_max_fit/const_c_km))
                            if np.any(cond_close_line) : 
                                cond_fit_order |= cond_close_line
                                dic_lines_order[sp][(i,j)] = copy.deepcopy(Analysis_dic['Lines']['Lines_fitted'][sp][(i,j)])
                                fitted_lines += 1

                    # Exclude pixels that should be excluded
                    if inst in settings['excluded_ranges'] : 
                        for plage in settings['excluded_ranges'][inst] : 
                            cond_fit_order &= ( (wl_ord < plage[0]) | (wl_ord > plage[1]) )

                    # Keep pixels that should be kept
                    if inst in settings['fitted_ranges'] : 
                        cond_in_at_least_one_plage = np.zeros_like(cond_fit_order)
                        for plage in settings['fitted_ranges'][inst] : 
                            cond_in_at_least_one_plage |= ( (wl_ord > plage[0]) & (wl_ord < plage[1]) )
                        cond_fit_order &= cond_in_at_least_one_plage

                    # All good !
                    dof += np.sum(cond_fit_order)

                    # We extract the pixels of Analysis_dic['table_wl_HR'] which cover the order
                    wl_min_ord, wl_max_ord = wl_ord[0], wl_ord[-1]
                    correspondance_HR = (Analysis_dic['table_wl_HR'] > wl_min_ord - 0.5) & (Analysis_dic['table_wl_HR'] < wl_max_ord + 0.5)

                    # If the LSF associated with the instrument is has a profile which is constant in wavelength, we store a new HR wl table with a constant wavelength step 
                    if Analysis_dic['LSF'][inst][date][spec]['mode'] == 'wl' : 
                        wl_HR_local = np.linspace( wl_min_ord - 0.25, wl_max_ord + 0.25, int( (wl_max_ord - wl_min_ord + 1)/Analysis_dic['LSF'][inst][date][spec]['step'] ) + 1   )
                    else : 
                        wl_HR_local = Analysis_dic['table_wl_HR'][correspondance_HR]

                    # Compute systematic uncertainties
                    #    + These uncerntainties are used to account for the simplicity of the model
                    #    + They are added quadratically to the tabulated erros     
                    sigma_model = np.zeros_like(wl_ord)
                    for dic in settings['sigma_model'] : 
                        if inst == dic['inst'] : 
                            cond = (wl_ord > dic['plage'][0]) & (wl_ord < dic['plage'][1])
                            p = np.poly1d(dic['coeff'])
                            sigma_model[cond] = p(1 - flux_renorm[cond])
                            sigma_model[cond & (1 - flux_renorm < 0)] = p(0)
                            sigma_model[cond & (1 - flux_renorm > 1)] = p(1)


                    # Store order data
                    Analysis_dic['Fitted_orders'][inst][date][spec][i_ord] = {'wl' : wl_ord,                                                   # WL pixels of the order
                                                                                'wl_edge' : def_edge_tab(wl_ord),                                # Edges of wl pixels
                                                                                'flux' : flux_renorm,                                            # Measured flux
                                                                                'error' : error_renorm,                                          # Measured error bars
                                                                                'sigma_model' : sigma_model,                                     # Systematic error bars
                                                                                'dic_lines' : dic_lines_order,                                   # dic of lines covered by the order
                                                                                'cond_fit_order' : cond_fit_order,                               # List of fitted pixels
                                                                                'correspondance_table_HR_global' : correspondance_HR,            # Part of the high-resolution grid which covers the orders
                                                                                'LSF_mode' : Analysis_dic['LSF'][inst][date][spec]['mode'],      # 'wl' or 'rv' (whether the LSF has a constant step in wavelength or RV)
                                                                                'wl_HR_local' : wl_HR_local,                                     # HR grid on which the instrumental LSF will be applied before resampling on the order pixels   
                                                                                'wl_HR_local_edge' : def_edge_tab(wl_HR_local),                  # Edges of wl_HR_local
                                                                    }


    print('Fitted lines :', fitted_lines, flush = True)
    print('Degrees of freedom :', dof, flush = True)

    Analysis_dic['dof'] = dof

    return None


# Retrieve the HFS properties of the Mn II triplet at 2600 A
def Retrieve_HFS(Analysis_dic, settings) : 

    Analysis_dic['HFS'] = {
        
        'Mn II' : {
                (0, 35) : [ [ 2606.478558846727 , 0.2856326763781777 ],     # Wavelength, weight.   Sum(weights) = 1
                            [ 2606.468640037975 , 0.04398743216223937 ],
                            [ 2606.4632730427966 , 0.19394458726078267 ],
                            [ 2606.4604876489484 , 0.003427592116538132 ],
                            [ 2606.455120687343 , 0.06455298486146817 ],
                            [ 2606.4509765928715 , 0.12253641816623823 ],
                            [ 2606.448802646943 , 0.008283347614967154 ],
                            [ 2606.4446585725614 , 0.06598114824335904 ],
                            [ 2606.441669412336 , 0.06855184233076264 ],
                            [ 2606.440174834794 , 0.011425307055127107 ],
                            [ 2606.437185684853 , 0.05427020851185376 ],
                            [ 2606.435419372201 , 0.02970579834333048 ],
                            [ 2606.4344682817646 , 0.010568409025992575 ],
                            [ 2606.432701972796 , 0.0371322479291631 ]],
                            
                            
                (0, 36) : [[ 2594.51251263333 , 0.25335697998479856 ],
                            [ 2594.5106951309053 , 0.03242969343805421 ],
                            [ 2594.502684687598 , 0.03242969343805421 ],
                            [ 2594.500867198942 , 0.1573346845705599 ],
                            [ 2594.499453598415 , 0.04864454015708132 ],
                            [ 2594.492789502392 , 0.04864454015708132 ],
                            [ 2594.4913759106666 , 0.09095515581454268 ],
                            [ 2594.4902315756362 , 0.051178109956929306 ],
                            [ 2594.4851157372573 , 0.051178109956929306 ],
                            [ 2594.483971407749 , 0.048897897137066125 ],
                            [ 2594.483163646351 , 0.042817329617430955 ],
                            [ 2594.4794614130465 , 0.04231061565746136 ],
                            [ 2594.4786536544566 , 0.02584241195844945 ],
                            [ 2594.478182462178 , 0.02634912591841905 ],
                            [ 2594.4760284425142 , 0.02634912591841905 ],
                            [ 2594.475489938157 , 0.02128198631872308 ]],
                
                
                
                (0, 37) : [[ 2576.8912790758054 , 0.25926886180969666 ],
                            [ 2576.8895525811095 , 0.025149079595540573 ],
                            [ 2576.8880252992685 , 0.0012963443090484833 ],
                            [ 2576.879857692325 , 0.19704433497536947 ],
                            [ 2576.878330421976 , 0.038371791547835106 ],
                            [ 2576.8771351681835 , 0.0025926886180969665 ],
                            [ 2576.8703620843003 , 0.14544983147523982 ],
                            [ 2576.8691668378997 , 0.041483017889551464 ],
                            [ 2576.8681708000795 , 0.003370495203526057 ],
                            [ 2576.862991415825 , 0.10396681358568836 ],
                            [ 2576.861995382779 , 0.036297640653357534 ],
                            [ 2576.8613313611754 , 0.0025926886180969665 ],
                            [ 2576.857612846522 , 0.07155820585947628 ],
                            [ 2576.856948827177 , 0.02385273528649209 ],
                            [ 2576.8542927532208 , 0.04770547057298418 ]]
                    
                            
                            },


    }

    return None





'''

Function used to calculate the model and compare to data

+ Uses the Analysis_dic, 
+ The model is computed from a simple array of parameters (params), as well as a few meta-parameters (meta-params), and the number of components (n_comp)

'''



# Function to calculate the excitation state of all the transiting components
#    + Each component is computed successively, starting from the component located closer to the star
#    + The transmitted flux in the saturated lines is stored iteratively, to account for the self-absorption and mutual occultation of the transiting exocomets.
def calc_abundances(Analysis_dic, params, meta_params, n_comp) :

    # List of studied species
    list_species = Analysis_dic['List_species']

    # Transform list of parameters into dictionary for easier use
    dic_params_all_components = list_to_dic(params, meta_params, list_species, n_comp)
    component_order = np.argsort([dic_params_all_components[i]['d'] for i in range(n_comp)])

    # Retrieve HR wavelength table on which the transmitted spectrum is calculated
    table_wl_HR = Analysis_dic['table_wl_HR']

    # Compute the line profiles of all the lines from all the components
    line_profiles_all_components = compute_line_profiles(Analysis_dic, dic_params_all_components)

    # Compute the escape probabilities functions
    escape_probabilities_all_components = compute_escape_probabilities(Analysis_dic, dic_params_all_components, meta_params, line_profiles_all_components)

    # Initialise transmitted flux and abundances
    abundances_all_components = {}
    transmitted_flux_total = np.ones_like(table_wl_HR)

    # Gaseous components are computed successively
    for i_comp in component_order : 
        transmitted_flux_component, abundances_component = calc_equilibrium_component(Analysis_dic, meta_params, transmitted_flux_total, dic_params_all_components, line_profiles_all_components, escape_probabilities_all_components, i_comp)
        abundances_all_components[i_comp] = abundances_component
        transmitted_flux_total *= transmitted_flux_component

    return dic_params_all_components, line_profiles_all_components, abundances_all_components


# Compute the full transmitted spectrum of the transiting exocomets
def calc_full_profile(Analysis_dic, abundances_all_components, dic_params_all_components, line_profiles_all_components) :

    # Retrieve table of wavelength close to strong lines
    table_wl_HR = Analysis_dic['table_wl_HR']

    transmitted_flux_all_components = {}

    # Flux is absorbed by exocomets and disc
    n_comp = len(abundances_all_components.keys())
    component_order = np.argsort([dic_params_all_components[i]['d'] for i in range(n_comp)])
    transmitted_flux_total = np.ones_like(table_wl_HR)
    for i_comp in component_order : 
        transmitted_flux_components = calc_transmitted_flux_component(Analysis_dic, table_wl_HR, Analysis_dic['Lines']['Lines_fitted'], dic_params_all_components, i_comp, abundances_all_components[i_comp], 1, line_profiles_all_components, use_alpha = True)
        transmitted_flux_all_components[i_comp] = transmitted_flux_components
        transmitted_flux_total *= transmitted_flux_components

    # Flux is absorbed by ISM
    transmitted_flux_ISM = calc_transmitted_flux_ISM(Analysis_dic, table_wl_HR) 
    transmitted_flux_all_components['ISM'] = transmitted_flux_ISM
    transmitted_flux_total *= transmitted_flux_ISM

    # All good
    transmitted_flux_all_components['Total'] = transmitted_flux_total

    return transmitted_flux_all_components


# Resample the transmitted spectrum on the instrumental wavelength tables
def resample_model_orders(Analysis_dic, transmitted_flux_all_components, n_comp, resample_components = True, dic_level_focus = None) : 
    dic_flux_all_orders = {}
    table_wl_HR = Analysis_dic['table_wl_HR']

# Iterate over instrument, date, spectrum, and order
    for inst in  Analysis_dic['Fitted_orders'] : 
        dic_flux_all_orders[inst] = {}    
        for date in Analysis_dic['Fitted_orders'][inst] : 
            dic_flux_all_orders[inst][date] = {}
            for spec in Analysis_dic['Fitted_orders'][inst][date] : 
                dic_flux_all_orders[inst][date][spec] = {}
                for i_ord in Analysis_dic['Fitted_orders'][inst][date][spec]:

                    dic_flux_all_orders[inst][date][spec][i_ord] = {}

                    # Pixel from Analysis_dic['table_wl_HR'] that overlap with current order
                    correspondance_ord_HR = Analysis_dic['Fitted_orders'][inst][date][spec][i_ord]['correspondance_table_HR_global']

                    wl_HR_local      =     Analysis_dic['Fitted_orders'][inst][date][spec][i_ord]['wl_HR_local']
                    wl_HR_local_edge =     Analysis_dic['Fitted_orders'][inst][date][spec][i_ord]['wl_HR_local_edge']

                    # Resample model flux to a new wavelength table with constant step in wl, if necessary
                    if Analysis_dic['Fitted_orders'][inst][date][spec][i_ord]['LSF_mode'] == 'wl' : 
                        model_HR_local = resampling(wl_HR_local_edge, def_edge_tab(table_wl_HR[correspondance_ord_HR]), transmitted_flux_all_components['Total'][correspondance_ord_HR], kind = 'cubic')  
                    else : 
                        model_HR_local = transmitted_flux_all_components['Total'][correspondance_ord_HR]

                    # Convolve with instrumental LSF
                    kernel = Analysis_dic['LSF'][inst][date][spec]['kernel']
                    model_HR_local_convolved = convolve(model_HR_local, kernel, mode = 'same')

                    # Resample on instrumental wavelength table
                    wl_LR_local =  Analysis_dic['Fitted_orders'][inst][date][spec][i_ord]['wl']
                    wl_LR_local_edge =  Analysis_dic['Fitted_orders'][inst][date][spec][i_ord]['wl_edge']
                    model_LR_local_convolved = resampling(wl_LR_local_edge, wl_HR_local_edge, model_HR_local_convolved, kind = 'cubic')   

                    # Store convoled and resampled data
                    dic_flux_all_orders[inst][date][spec][i_ord]['wl_ord_LR']      = wl_LR_local
                    dic_flux_all_orders[inst][date][spec][i_ord]['Flux_obs_LR']    = Analysis_dic['Fitted_orders'][inst][date][spec][i_ord]['flux']
                    dic_flux_all_orders[inst][date][spec][i_ord]['Error_obs_LR']   = Analysis_dic['Fitted_orders'][inst][date][spec][i_ord]['error']
                    dic_flux_all_orders[inst][date][spec][i_ord]['cond_fit_LR']    = Analysis_dic['Fitted_orders'][inst][date][spec][i_ord]['cond_fit_order']
                    dic_flux_all_orders[inst][date][spec][i_ord]['sigma_model_LR'] = Analysis_dic['Fitted_orders'][inst][date][spec][i_ord]['sigma_model']
                    dic_flux_all_orders[inst][date][spec][i_ord]['Flux_model_LR']  = model_LR_local_convolved

                    # Store the HR, unconvolved fluxes transmitted by each exocomet and by the ISM. In addition, store wl tables to simplify the plots
                    cond_kept_HR = (wl_HR_local>wl_LR_local[0]) & (wl_HR_local<wl_LR_local[-1])

                    # Iterate over exocomets
                    for i_comp in range(n_comp) :
                        if Analysis_dic['Fitted_orders'][inst][date][spec][i_ord]['LSF_mode'] == 'wl' : 
                            model_HR_local_comp = resampling(wl_HR_local_edge, def_edge_tab(table_wl_HR[correspondance_ord_HR]), transmitted_flux_all_components[i_comp][correspondance_ord_HR], kind = 'cubic')  
                        else : 
                            model_HR_local_comp = transmitted_flux_all_components[i_comp][correspondance_ord_HR]
                        dic_flux_all_orders[inst][date][spec][i_ord]['Flux_comp_'+str(i_comp)+'_HR'] = model_HR_local_comp[cond_kept_HR]

                    # ISM
                    if Analysis_dic['Fitted_orders'][inst][date][spec][i_ord]['LSF_mode'] == 'wl' : 
                        model_HR_local_ISM = resampling(wl_HR_local_edge, def_edge_tab(table_wl_HR[correspondance_ord_HR]), transmitted_flux_all_components['ISM'][correspondance_ord_HR], kind = 'cubic')  
                    else : 
                        model_HR_local_ISM = transmitted_flux_all_components['ISM'][correspondance_ord_HR]
                    dic_flux_all_orders[inst][date][spec][i_ord]['Flux_ISM_HR'] = model_HR_local_ISM[cond_kept_HR]

                    # Full model
                    dic_flux_all_orders[inst][date][spec][i_ord]['wl_ord_HR'] = wl_HR_local[cond_kept_HR]
                    dic_flux_all_orders[inst][date][spec][i_ord]['Flux_model_HR'] = model_HR_local[cond_kept_HR]
                    dic_flux_all_orders[inst][date][spec][i_ord]['Flux_model_HR_convolved'] = model_HR_local_convolved[cond_kept_HR]

    return dic_flux_all_orders


# Compare the model to the data by calculating the Chi2
def calc_chi2(dic_flux_all_orders) :

    chi2, dof = 0, 0
    for inst in  dic_flux_all_orders : 
        for date in  dic_flux_all_orders[inst] : 
            for spec in dic_flux_all_orders[inst][date] : 
                for i_ord in dic_flux_all_orders[inst][date][spec]:
                    cond_fit_order = dic_flux_all_orders[inst][date][spec][i_ord]['cond_fit_LR']
                    chi2 += np.sum(  (dic_flux_all_orders[inst][date][spec][i_ord]['Flux_obs_LR'][cond_fit_order] - dic_flux_all_orders[inst][date][spec][i_ord]['Flux_model_LR'][cond_fit_order])**2 / (dic_flux_all_orders[inst][date][spec][i_ord]['Error_obs_LR'][cond_fit_order]**2   + dic_flux_all_orders[inst][date][spec][i_ord]['sigma_model_LR'][cond_fit_order]**2  )   ) 
                    dof += np.sum(cond_fit_order)

    return chi2, dof


# Convert list of parameters into dic
def list_to_dic(list_params, meta_params, l_species, n_comp) : 

    dic_params = {}
    n_species = len(l_species)

    i_start_comp = 0

    for i_comp in range(n_comp) :
        n_pixels = meta_params[i_comp]['n_pix']
        n_params_comp = 7 + 3*n_pixels + n_species
        dic_params[i_comp] = {}

        dic_params[i_comp]['d']       = list_params[i_start_comp + 0]
        dic_params[i_comp]['Delta_d'] = list_params[i_start_comp + 1]
        dic_params[i_comp]['f_esc']   = list_params[i_start_comp + 2]
        dic_params[i_comp]['xi']      = list_params[i_start_comp + 3]
        dic_params[i_comp]['T_i'] = list_params[i_start_comp + 4]
        dic_params[i_comp]['log_n_e'] = list_params[i_start_comp + 5]
        dic_params[i_comp]['log_T_e'] = list_params[i_start_comp + 6]

        dic_params[i_comp]['v'] = []
        dic_params[i_comp]['alpha'] = []
        dic_params[i_comp]['phi_macro'] = []
        for i in range(n_pixels) :
            dic_params[i_comp]['v']        .append(list_params[i_start_comp + 7 + 3*i + 0])    
            dic_params[i_comp]['alpha']    .append(list_params[i_start_comp + 7 + 3*i + 1])    
            dic_params[i_comp]['phi_macro'].append(list_params[i_start_comp + 7 + 3*i + 2])    
        
        dic_params[i_comp]['N'] = {}
        for i_sp, sp in enumerate(l_species) : 
            dic_params[i_comp]['N'][sp] = list_params[i_start_comp + 7 + 3*n_pixels + i_sp]

        i_start_comp += n_params_comp
        
    return dic_params


# Convert dic of parameters into a list
def dic_to_list(dic_params, meta_params, l_species) : 

    list_params = []
    for i_comp in dic_params : 
        n_pixels = meta_params[i_comp]['n_pix']
        for key in ['d', 'Delta_d', 'f_esc', 'xi', 'T_i', 'log_n_e', 'log_T_e'] : list_params.append(dic_params[i_comp][key])
        for i in range(n_pixels) : 
            list_params.append(dic_params[i_comp]['v'][i])
            list_params.append(dic_params[i_comp]['alpha'][i])
            list_params.append(dic_params[i_comp]['phi_macro'][i])
        for sp in l_species : 
            list_params.append(dic_params[i_comp]['N'][sp])
    return list_params


# Function to build a continuous profile for the covering factor (alpha) of a given component
def make_profile_alpha(makima, list_v_fit) : 

    def func(v) : 
        result = np.zeros_like(v)
        result[(v > np.min(list_v_fit)) & (v < np.max(list_v_fit))] = makima(v[(v > np.min(list_v_fit)) & (v < np.max(list_v_fit))])
        result[v < np.min(list_v_fit)] = makima(np.min(list_v_fit))
        result[v > np.max(list_v_fit)] = makima(np.max(list_v_fit))
        return result

    return func


# Function to build a continuous line profile (phi) for a given component
def make_profile_phi(makima, list_v_fit) : 

    def func(v) : 
        result = np.zeros_like(v)
        result[(v > np.min(list_v_fit)) & (v < np.max(list_v_fit))] = makima(v[(v > np.min(list_v_fit)) & (v < np.max(list_v_fit))])
        result[v < np.min(list_v_fit)] = 0
        result[v > np.max(list_v_fit)] = 0
        return result

    return func


# Compute the profiles (covering factor and line profile) of all the studied components
def compute_line_profiles(Analysis_dic, dic_params_all_components) :

    line_profiles_all_components = {}
    for i_comp in dic_params_all_components : 

        line_profiles_all_components[i_comp] = {}

        # Retrieve component parameters
        xi = dic_params_all_components[i_comp]['xi']
        T_i = dic_params_all_components[i_comp]['T_i']
        list_v = dic_params_all_components[i_comp]['v']
        list_phi_macro = dic_params_all_components[i_comp]['phi_macro']
        list_alpha = dic_params_all_components[i_comp]['alpha']

        # Identify profile mode
        if len(list_v) == 1 : mode = 'gauss'
        if len(list_v) >= 2 and xi == 0 and T_i == 0 : mode = 'spline'
        if len(list_v) >= 2 and (xi != 0 or T_i != 0): mode = 'spline_gauss'
        
        # Retrieve alpha and column density profiles
        if mode == 'gauss' : 
            alpha = dic_params_all_components[i_comp]['alpha'][0]

        if mode in ['spline',  'spline_gauss'] : 
            list_v_fit = np.array([list_v[0] - (list_v[1] - list_v[0])] + list_v + [list_v[-1] + (list_v[-1] - list_v[-2])])
            list_alpha_fit   = np.array([list_alpha[0]] + list_alpha + [list_alpha[-1]])
            list_phi_macro_fit = np.array([0] + list_phi_macro + [0])
                        
            spline_alpha_temp     = PchipInterpolator(list_v_fit, list_alpha_fit)
            spline_phi_macro_temp = PchipInterpolator(list_v_fit, list_phi_macro_fit)    
            spline_alpha = make_profile_alpha(spline_alpha_temp, list_v_fit)
            spline_phi_macro = make_profile_phi(spline_phi_macro_temp, list_v_fit)

        # Recompute the profile for each studied line
        for sp in Analysis_dic['List_species'] : 
            line_profiles_all_components[i_comp][sp] = {}

            b_gauss_rv = np.sqrt(  xi**2    +    2e-6*const_k_b*T_i / (give_mass(sp)*const_m_H)  )

            # Compute a generic rv profile on a high-resolution RV table (0.1 km/s)
            #    + The lorentzian wings will be added later if necessary
            rv_generic = np.linspace(-100,100,int(200/0.1)+1)
            rv_generic_edge = def_edge_tab(rv_generic)
            d_rv_generic = 0.1

            if mode == 'gauss' : 
                phi_comp  = np.exp(   -(rv_generic - list_v[0])**2/b_gauss_rv**2   )

            if mode in ['spline',  'spline_gauss'] : 
                phi_comp_macro = spline_phi_macro(rv_generic)
                phi_comp_macro[rv_generic<list_v_fit[0]] = 0 
                phi_comp_macro[rv_generic>list_v_fit[-1]] = 0 
                phi_comp_macro[phi_comp_macro<0] = 0 

                if mode == 'spline_gauss' : 
                    phi_comp_gauss  = np.exp(   -rv_generic**2/b_gauss_rv**2   )
                    phi_comp = convolve(phi_comp_macro, phi_comp_gauss, mode = 'same')

                if mode == 'spline' : phi_comp = phi_comp_macro

            phi_comp /= np.sum(phi_comp * d_rv_generic)     # (km/s)-1

            line_profiles_all_components[i_comp][sp]['Profile'] = {'rv' : rv_generic, 'rv_edge' : rv_generic_edge, 'phi_rv' : phi_comp}

            
            for (i,j) in  set(Analysis_dic['Lines']['Lines_fitted'][sp].keys()) | set(Analysis_dic['Lines']['Lines_saturated'][sp].keys())  :

                # Retrieve line properties
                if (i,j) in  Analysis_dic['Lines']['Lines_fitted'][sp] : 
                    wl =  Analysis_dic['Lines']['Lines_fitted'][sp][(i,j)]['wl']
                    b_nat_rv = Analysis_dic['Lines']['Lines_fitted'][sp][(i,j)]['b_nat_wl']/wl * const_c_km
                    rv_close_HR = Analysis_dic['Lines']['Lines_fitted'][sp][(i,j)]['rv_close_HR'] 
                    rv_close_HR_edge = Analysis_dic['Lines']['Lines_fitted'][sp][(i,j)]['rv_close_HR_edge'] 
                else : 
                    wl =  Analysis_dic['Lines']['Lines_saturated'][sp][(i,j)]['wl']
                    b_nat_rv = Analysis_dic['Lines']['Lines_saturated'][sp][(i,j)]['b_nat_wl']/wl * const_c_km  
                    rv_close_HR = Analysis_dic['Lines']['Lines_saturated'][sp][(i,j)]['rv_close_HR'] 
                    rv_close_HR_edge = Analysis_dic['Lines']['Lines_saturated'][sp][(i,j)]['rv_close_HR_edge'] 

                d_rv_close_HR = np.diff(rv_close_HR_edge)

                # If the line has an HFS, the profile is recalculated
                if sp in Analysis_dic['HFS'] and (i,j) in Analysis_dic['HFS'][sp] :

                    phi_line = np.zeros_like(rv_generic)
                    for sub_wl, weight in Analysis_dic['HFS'][sp][(i,j)] : 
                        delta_rv = (sub_wl - wl)/wl * const_c_km

                        if mode == 'gauss' : 
                            phi_sub_line = np.exp(   -(rv_generic - delta_rv - list_v[0])**2/b_gauss_rv**2   )

                        if mode in ['spline',  'spline_gauss'] : 

                            phi_sub_line_macro = spline_phi_macro(rv_generic - delta_rv)
                            phi_sub_line_macro[rv_generic - delta_rv < list_v_fit[0]]  = 0 
                            phi_sub_line_macro[rv_generic - delta_rv > list_v_fit[-1]] = 0 
                            phi_sub_line_macro[phi_sub_line_macro<0] = 0 

                            if mode == 'spline_gauss' : 
                                phi_sub_line_gauss = np.exp(   -rv_generic**2/b_gauss_rv**2   )
                                phi_sub_line = convolve(phi_sub_line_macro, phi_sub_line_gauss, mode = 'same')

                            if mode == 'spline' : 
                                phi_sub_line = phi_sub_line_macro

                        phi_line += phi_sub_line / np.sum(phi_sub_line) * weight

                    phi_line = resampling(rv_close_HR_edge, rv_generic_edge, phi_line)

                # Otherwise, we resample the line profile of the component on the wavelength pixels around the line
                else : phi_line = resampling(rv_close_HR_edge, rv_generic_edge, phi_comp)

                # If the line is strong (low excitation energy, strong oscillator strength): convolution with a lorentzian profile 
                if b_nat_rv > 0.0001 and i < 3 and sp in ['Fe II', 'Si II', 'C I 3P 1/2', 'C I 3P 3/2', 'C I 3P 5/2'] : 

                    phi_loren = lorentzian_average(rv_close_HR, d_rv_close_HR, b_nat_rv)
                    phi_line = convolve(phi_line, phi_loren, mode = 'same')

                # The profile is renormalised so that integral(profile * d_rv) = 1
                phi_line /= np.sum(phi_line * d_rv_close_HR)
                phi_line_sum1 = phi_line/np.sum(phi_line)

                # Also extract the alpha profile of the line
                if mode in ['spline',  'spline_gauss'] :
                    profile_alpha = spline_alpha(rv_close_HR)
                    profile_alpha[rv_close_HR < list_v_fit[0 ]] = list_alpha_fit[0 ]
                    profile_alpha[rv_close_HR > list_v_fit[-1]] = list_alpha_fit[-1]

                # Store line profile
                if mode == 'gauss'                    : line_profiles_all_components[i_comp][sp][(i,j)] = {'phi_rv' : phi_line,  'phi_sum1' : phi_line_sum1,   'alpha' : alpha}
                if mode in ['spline', 'spline_gauss'] : line_profiles_all_components[i_comp][sp][(i,j)] = {'phi_rv' : phi_line,  'phi_sum1' : phi_line_sum1,   'alpha' : profile_alpha}
                

    return line_profiles_all_components


# Build a function returning the escape probability in a given line, as a function of N (in 10^14 cm-2) * f * lambda/2000 A
def make_function_escape_probability(tab_log_N_1e14_f_wl_2000, tab_log_p_esc) : 
    spline_log_N_log_p_esc = CubicSpline(tab_log_N_1e14_f_wl_2000, tab_log_p_esc, bc_type = 'natural')
    def func(N) :
        if N < 1e-6 : return 1
        else : return 10**spline_log_N_log_p_esc(np.log10(N))
    return func


# Pre-compute the escape probabilities of photon emitted through spontaneous emission
def compute_escape_probabilities(Analysis_dic, dic_params_all_components, meta_params, line_profiles_all_components) : 

    dic_escape_probabilities = {}

    for i_comp in dic_params_all_components : 
        dic_escape_probabilities[i_comp] = {}
        for sp in Analysis_dic['List_species'] : 

            # Retrieve the line profile of component and species
            rv =  line_profiles_all_components[i_comp][sp]['Profile']['rv']        
            phi_rv =  line_profiles_all_components[i_comp][sp]['Profile']['phi_rv']        
            d_rv = rv[1] - rv[0]

            # Exact escape probability
            def p_esc_exact(N_1e14_f_wl_2000) : 
                tau = 530.802 * phi_rv * N_1e14_f_wl_2000

                theta = np.linspace(0, (pi/2) - 0.001, 31)

                d_theta = theta[1] - theta[0]

                if meta_params[i_comp]['geometry'] == 'slab' : phi_theta = 2 * pi * np.sin(theta)    # Slab geometry
                if meta_params[i_comp]['geometry'] == 'cylind' : phi_theta = 2 * pi * np.cos(theta)  # Cylindrical geometry
                
                phi_theta /= np.sum(phi_theta*d_theta)    # (km/s)^-1

                p_esc_theta = [np.sum(phi_rv * d_rv * np.exp(-tau/np.cos(t))) for t in theta]
                p_esc = np.sum(p_esc_theta * phi_theta * d_theta)
                return p_esc 
            
            # Interpolate over a low-resolution grid 
            tab_log_N_1e14_f_wl_2000 = np.linspace(-15,10,51)
            tab_p_esc = np.array([p_esc_exact(10**x) for x in tab_log_N_1e14_f_wl_2000])
            tab_p_esc[tab_p_esc<1e-30] = 1e-30
            tab_log_p_esc = np.log10(tab_p_esc)

            dic_escape_probabilities[i_comp][sp] = make_function_escape_probability(tab_log_N_1e14_f_wl_2000, tab_log_p_esc)

    return dic_escape_probabilities


# Compute the excitation state along 1 component
#    + The component is divided in several gas bins to account for self-absorption
#    + Both collisional and radiative process are taken into account
def calc_equilibrium_component(Analysis_dic, meta_params, incoming_flux, dic_params_all_components, line_profiles_all_components, escape_probabilities_all_components, i_comp) :

    current_flux = copy.deepcopy(incoming_flux)

    d_N_bin = meta_params[i_comp]['d_N_bin']
    sp_ref = meta_params[i_comp]['sp_ref']
    frac_step = d_N_bin/dic_params_all_components[i_comp]['N'][sp_ref]
    initial_d = dic_params_all_components[i_comp]['d']
    Delta_d = dic_params_all_components[i_comp]['Delta_d']

    dic_Mat_radiation = {}
    dic_Mat_collision = {}
    dic_abundances_current = {}
    dic_abundances_total   = {}

    table_wl_HR = Analysis_dic['table_wl_HR']

    # Initialize the excitation state of each species
    for sp in Analysis_dic['List_species'] : 
        n_level_modeled_sp = Analysis_dic['Data_species'][sp]['n_level_modeled']

        if n_level_modeled_sp >= 2 : 
            Mat_rad = np.zeros((n_level_modeled_sp, n_level_modeled_sp))
            abundances_current = np.zeros(n_level_modeled_sp)

            Mat_col = collision_matrix(Analysis_dic, sp, dic_params_all_components, i_comp)

            n_step_init = 5
            for i in range(n_step_init) : 
                Mat_rad = radiation_matrix(Analysis_dic, sp, current_flux,  dic_params_all_components, initial_d, i_comp, Mat_rad, line_profiles_all_components, escape_probabilities_all_components, update_only_strong = (i>=1), abundances_guess_sp = abundances_current*i/n_step_init)
                abundances_current = solve_linear_system(Mat_rad + Mat_col)

            dic_Mat_collision[sp] = Mat_col
            dic_Mat_radiation[sp] = Mat_rad
            dic_abundances_current[sp] = abundances_current
            dic_abundances_total[sp] = np.zeros(n_level_modeled_sp)

        if n_level_modeled_sp == 1 : 
            dic_abundances_current[sp] = np.array([1.])
            dic_Mat_collision[sp] = None
            dic_Mat_radiation[sp] = None
            dic_abundances_total[sp] = np.zeros(n_level_modeled_sp)

    # Calculate the excitation of the gas and the absorbed flux, step by step
    i_step = 0 
    remaining_fraction = 1
    keep_calculate = True   
    while keep_calculate :
        i_step += 1

        if remaining_fraction > frac_step + 1e-10 : 
            frac_step_loc = frac_step
            remaining_fraction -= frac_step
        else : 
            frac_step_loc = remaining_fraction
            keep_calculate = False

        for sp in Analysis_dic['List_species'] :  
            n_level_modeled_sp = Analysis_dic['Data_species'][sp]['n_level_modeled']

            if n_level_modeled_sp >= 2 : 
                dic_Mat_radiation[sp] = radiation_matrix(Analysis_dic, sp, current_flux, dic_params_all_components, initial_d + Delta_d*(1-remaining_fraction), i_comp, dic_Mat_radiation[sp], line_profiles_all_components, escape_probabilities_all_components, update_only_strong = True, abundances_guess_sp = dic_abundances_current[sp]) 
                abundances_current = solve_linear_system(dic_Mat_radiation[sp] + dic_Mat_collision[sp])
                dic_abundances_current[sp] = abundances_current

            dic_abundances_total[sp] += dic_abundances_current[sp]*frac_step_loc

        current_flux *= calc_transmitted_flux_component(Analysis_dic, table_wl_HR, Analysis_dic['Lines']['Lines_saturated'], dic_params_all_components, i_comp, dic_abundances_current, frac_step_loc, line_profiles_all_components, use_alpha = False)

    for sp in dic_abundances_total : dic_abundances_total[sp] /= np.sum(dic_abundances_total[sp])
    flux_transmitted_component = calc_transmitted_flux_component(Analysis_dic, table_wl_HR, Analysis_dic['Lines']['Lines_saturated'], dic_params_all_components, i_comp, dic_abundances_total, 1,  line_profiles_all_components, use_alpha = True)
    
    return flux_transmitted_component, dic_abundances_total


# Build the matrix of all radiative exchanges between excitation states of a given species (sp)
#    + The population fluxes depend on the received stellar flux, potentially atenuated by exocomet absorption
#    + If the matrix has already been calculated, we only update the saturated lines (for which the stellar flux decreases as the star gets more and pore occulted by the gas). 
def radiation_matrix(Analysis_dic, sp, incoming_flux, dic_params_all_components, d, i_comp, Mat, line_profiles_all_components, escape_probabilities_all_components, update_only_strong = False, abundances_guess_sp = None) : 

    coef_mat = const_c**2/(8*pi*const_h)

    # Transverse column density of the cloud
    N_esc =  dic_params_all_components[i_comp]['f_esc'] *  dic_params_all_components[i_comp]['N'][sp]

    # Dilution factor
    x = 1/(d*AU_R_beta_pic)
    W = ( 1 - np.sqrt(1-x**2) ) * 2*pi

    dic_lines_saturated = Analysis_dic['Lines']['Lines_saturated'][sp]
    n_level_modeled = Analysis_dic['Data_species'][sp]['n_level_modeled']
    tab_A = Analysis_dic['Data_species'][sp]['A']
    tab_mult = Analysis_dic['Data_species'][sp]['mult']
    tab_nu = Analysis_dic['Data_species'][sp]['nu']
    I = Analysis_dic['Reference_spectrum']['Specific_intensity_lines'][sp]

    # Table of the attenuation factors due to the self-absorption
    tab_fact = np.ones((n_level_modeled, n_level_modeled)) * W
    for (i,j) in dic_lines_saturated:   
        if j < n_level_modeled : 
            ind_close_line = dic_lines_saturated[(i,j)]['ind_close_HR']
            kernel = line_profiles_all_components[i_comp][sp][(i,j)]['phi_sum1']
            flux_line = incoming_flux[ind_close_line]
            fact = np.sum(    flux_line * kernel   )
            tab_fact[i,j] *= fact
            tab_fact[j,i] *= fact

    # Table of escape probability 
    tab_esc = np.ones((n_level_modeled, n_level_modeled))
    if abundances_guess_sp is not None : 
        for (i,j) in dic_lines_saturated:  
            wl = dic_lines_saturated[(i,j)]['wl']
            f  = dic_lines_saturated[(i,j)]['f']
            tab_esc[i,j] = escape_probabilities_all_components[i_comp][sp](N_esc * abundances_guess_sp[i] * wl/2000 * f)
            tab_esc[j,i] = escape_probabilities_all_components[i_comp][sp](N_esc * abundances_guess_sp[i] * wl/2000 * f)

    # Coefficients i != j
    if update_only_strong : 
        for (i,j) in dic_lines_saturated: 
            Mat[i,j] = tab_A[i,j] * (tab_esc[i,j] + coef_mat/tab_nu[i,j]**3 * I[i,j] * tab_fact[i,j])
            Mat[j,i] = tab_A[i,j] * tab_mult[j]/tab_mult[i] * coef_mat/tab_nu[i,j]**3 * I[i,j] * tab_fact[i,j]

    else :
        dic_lines_all = Analysis_dic['Lines']['Lines_all'][sp]
        for (i,j) in dic_lines_all: 
            Mat[i,j] = tab_A[i,j] * (tab_esc[i,j] + coef_mat/tab_nu[i,j]**3 * I[i,j] * tab_fact[i,j])
            Mat[j,i] = tab_A[i,j] * tab_mult[j]/tab_mult[i] * coef_mat/tab_nu[i,j]**3 * I[i,j] * tab_fact[i,j]

    # Coefficients i = j
    if update_only_strong :  
        list_levels_saturated = Analysis_dic['Lines']['Levels_saturated'][sp]
        for i in list_levels_saturated :
            Mat[i,i] = 0
            Mat[i,i] = -np.sum(Mat[:,i])

    else :  
        for i in range(n_level_modeled):
            Mat[i,i] = 0
            Mat[i,i] = -np.sum(Mat[:,i])


    return Mat


# Build the matrix of all colisionnal exchanges between excitation states of a given species (sp)
def collision_matrix(Analysis_dic, sp, dic_params_all_components, i_comp) : 

    n_level_modeled = Analysis_dic['Data_species'][sp]['n_level_modeled']
    tab_mult = Analysis_dic['Data_species'][sp]['mult']
    tab_delta_E_K = Analysis_dic['Data_species'][sp]['delta_E_K']
    dic_spline_upsilon = Analysis_dic['Data_species'][sp]['spline_upsilon']
    Mat = np.zeros((n_level_modeled, n_level_modeled))
    log_n_e = dic_params_all_components[i_comp]['log_n_e']
    n_e = 10**log_n_e
    log_T_e = dic_params_all_components[i_comp]['log_T_e']
    T_e = 10**log_T_e

    for i in range(n_level_modeled) : 
        for j in range(n_level_modeled) : 
            if i!=j and dic_spline_upsilon[(i,j)] is not None : 
                upsilon = dic_spline_upsilon[(i,j)](log_T_e)
                if j < i : 
                    C_ji = 10**6 * const_Beta_col/np.sqrt(T_e) * upsilon/tab_mult[j] * np.exp(-tab_delta_E_K[i,j]/T_e)
                    Mat[i,j] += C_ji*n_e
                if j > i :
                    C_ji = 10**6 * const_Beta_col/np.sqrt(T_e) * upsilon/tab_mult[j]
                    Mat[i,j] += C_ji*n_e
            if j == i : 
                for k in range(n_level_modeled) : 
                    if dic_spline_upsilon[(i,k)] is not None :
                        upsilon = dic_spline_upsilon[(i,k)](log_T_e)

                        if k < i : 
                            C_ik = 10**6 * const_Beta_col/np.sqrt(T_e) * upsilon/tab_mult[i]
                            Mat[i,j] -= C_ik*n_e

                        if k > i : 
                            C_ik = 10**6 * const_Beta_col/np.sqrt(T_e) * upsilon/tab_mult[i] * np.exp(-tab_delta_E_K[i,k]/T_e)
                            Mat[i,j] -= C_ik*n_e

    return Mat


# Solve AX = B to compute the statistical equilibrium of the gas
def solve_linear_system(Mat) : 
    A = copy.deepcopy(Mat)
    A[0] = np.ones(len(A[0]))
    B = np.zeros(len(A[0]))
    B[0] = 1    
    abundances = spsolve(csc_matrix(A), B)

    return abundances/np.sum(abundances)


# Compute the flux transmitted by a gaseous cloud
#    + The list of considered lines is given through dic_lines
#    + The column densities are passed through dic_params_all_components
#    + The excitation state is given is dic_abundance_levels
#    + The line profiles have been calculated in line_profiles_all_components
#    + The covering factor can be accounted for (when computing the flux transmitted by a whole component) or not (when computing the flux transmitted between separate gas bins of a given component)
def calc_transmitted_flux_component(Analysis_dic, table_wl,  dic_lines, dic_params_all_components, i_comp, dic_abundance_levels, frac_step, line_profiles_all_components, use_alpha = True) :

    flux = np.ones(len(table_wl))

    for sp in dic_lines : 
        for (i,j) in dic_lines[sp] : 

            phi_rv = line_profiles_all_components[i_comp][sp][(i,j)]['phi_rv']      # cm-2 / (km/s)
            wl, f = dic_lines[sp][(i,j)]['wl'],  dic_lines[sp][(i,j)]['f']

            ind_close_HR = dic_lines[sp][(i,j)]['ind_close_HR']
            tau = 5.30802 * (wl/2000) * (phi_rv * dic_params_all_components[i_comp]['N'][sp] * dic_abundance_levels[sp][i] / 0.01) * f * frac_step


            if use_alpha : 
                if len(dic_params_all_components[i_comp]['v']) >= 2 : 
                    alpha = line_profiles_all_components[i_comp][sp][(i,j)]['alpha']
                else : 
                    alpha = dic_params_all_components[i_comp]['alpha'][0]
            else : alpha = 1

            flux[ind_close_HR] *= alpha*np.exp(-tau) + 1-alpha

    return flux
        

# Compute the flux transmitted by the ISM
def calc_transmitted_flux_ISM(Analysis_dic, table_wl) :

    tau = np.zeros(len(table_wl))

    for sp in Analysis_dic['List_species'] : 
        if sp in Analysis_dic['ISM'] : 
            N = Analysis_dic['ISM'][sp]['N']
            b_gauss_rv = Analysis_dic['ISM'][sp]['b']
            v = Analysis_dic['ISM'][sp]['v']

            for (i,j) in Analysis_dic['Lines']['Lines_fitted'][sp] : 
                if i == 0 : 
                    wl, f = Analysis_dic['Lines']['Lines_fitted'][sp][(i,j)]['wl'], Analysis_dic['Lines']['Lines_fitted'][sp][(i,j)]['f']
                    b_nat_rv = Analysis_dic['Lines']['Lines_fitted'][sp][(i,j)]['b_nat_wl'] / wl * const_c_km

                    ind_close_HR = Analysis_dic['Lines']['Lines_fitted'][sp][(i,j)]['ind_close_HR']
                    wl_close_HR = Analysis_dic['table_wl_HR'][   ind_close_HR   ]
                    rv_HR = (wl_close_HR - wl)/wl * const_c_km

                    phi = voigt_profile(rv_HR - v, b_gauss_rv/np.sqrt(2), b_nat_rv)     # cm-2 / (km/s)-1

                    d_rv = np.diff(def_edge_tab(rv_HR))
                    phi /= np.sum(phi*d_rv)

                    tau[ind_close_HR] += 5.30802 * (wl/2000) * (N * phi/0.01) * f

    return np.exp(-tau)






'''

Plotting functions

'''






def apply_alpha_on_white(color, alpha):
    r, g, b = mplcolors.to_rgb(color)
    r2 = (1 - alpha) + alpha * r
    g2 = (1 - alpha) + alpha * g
    b2 = (1 - alpha) + alpha * b
    return f'rgb({int(r2*255)}, {int(g2*255)}, {int(b2*255)})'


def filter_nan(a, cond): 
    a_filtered = []
    prev_idx = None

    for idx in np.where(cond)[0]:
        if prev_idx is not None and idx != prev_idx + 1:
            a_filtered.append(np.nan)   # gap
        a_filtered.append(a[idx])
        prev_idx = idx

    return np.array(a_filtered)




#Function to plot observations: 
#   + Overlay spectra from different epochs with different colors
#   + No model can be shown
#   + Plot flux vs RV by settings line = (wavelength of your favorite line, in A)
#   + Show a given reference spectrum with plot_reference = '...'
def plot_observed_spectrum(spec_dic, inst, visits_plots, plot_error_bar=False, plot_reference=None,
               visits_emphasis=[], date_to_color={},
               xlim=None, ylim=None, line=None, title = None, loc_label = 'upper right'):

    fig = go.Figure()
    dates_in_legend = set()

    min_wl, max_wl = np.inf, -np.inf
    min_x,  max_x  = np.inf, -np.inf


    legend_anchor_map = {
    'best':         {},   # dict vide = Plotly gère tout seul
    'upper right':  dict(x=0.99, y=0.99, xanchor='right',  yanchor='top'),
    'upper left':   dict(x=0.01, y=0.99, xanchor='left',   yanchor='top'),
    'lower right':  dict(x=0.99, y=0.01, xanchor='right',  yanchor='bottom'),
    'lower left':   dict(x=0.01, y=0.01, xanchor='left',   yanchor='bottom'),
    'upper center': dict(x=0.5,  y=0.99, xanchor='center', yanchor='top'),
    'lower center': dict(x=0.5,  y=0.01, xanchor='center', yanchor='bottom'),
    }
    legend_pos = legend_anchor_map.get(loc_label, {})


    # Spectra which are not emphasized
    for date in visits_plots:
        if date in visits_emphasis: continue
        for spec in spec_dic['spec_renorm'][inst][date]:
            for i_ord in range(len(spec_dic['spec_renorm'][inst][date][spec])):
                wl   = spec_dic['spec_renorm'][inst][date][spec][i_ord]['wl']
                flux = spec_dic['spec_renorm'][inst][date][spec][i_ord]['flux']

                if line is not None : x = (wl - line) / line * const_c_km
                else                : x = wl

                y = copy.deepcopy(flux)
                if xlim is not None: cond_plot = ((x >= xlim[0]) & (x <= xlim[1]))
                else               : cond_plot = np.ones_like(x, dtype = bool)

                # Récupère la couleur de base, avec fallback sur un gris neutre
                base_color = date_to_color.get(date, '#6464C8')
                color = apply_alpha_on_white(base_color, 0.3)
                
                error_y = None
                if plot_error_bar:
                    yerr = spec_dic['spec_renorm'][inst][date][spec][i_ord]['error']
                    error_y = dict(type='data', array=filter_nan(yerr, cond_plot), visible=True,
                                   color=color, thickness=3)

                if np.any(cond_plot) : 
                    fig.add_trace(go.Scatter(
                        x=filter_nan(x, cond_plot), y=filter_nan(y, cond_plot),
                        mode='lines',
                        name=date,
                        legendgroup=date,
                        showlegend=False,
                        line=dict(color=color, width=1.2),
                        error_y=error_y,
                    ))

                    min_wl = min(min_wl, np.nanmin(wl[cond_plot]))
                    max_wl = max(max_wl, np.nanmax(wl[cond_plot]))

                    min_x = min(min_x, np.nanmin(x[cond_plot]))
                    max_x = max(max_x, np.nanmax(x[cond_plot]))



    # Spectra which are emphasized
    for date in visits_plots:
        if date not in visits_emphasis: continue
        for spec in spec_dic['spec_renorm'][inst][date]:
            for i_ord in range(len(spec_dic['spec_renorm'][inst][date][spec])):
                wl   = spec_dic['spec_renorm'][inst][date][spec][i_ord]['wl']
                flux = spec_dic['spec_renorm'][inst][date][spec][i_ord]['flux']

                if line is not None : x = (wl - line) / line * const_c_km
                else                : x = wl

                y = copy.deepcopy(flux)
                if xlim is not None: cond_plot = ((x >= xlim[0]) & (x <= xlim[1]))
                else               : cond_plot = np.ones_like(x, dtype = bool)

                color = date_to_color.get(date, '#6464C8')
                error_y = None
                if plot_error_bar:
                    yerr = spec_dic['spec_renorm'][inst][date][spec][i_ord]['error']
                    error_y = dict(type='data', array=filter_nan(yerr, cond_plot), visible=True,
                                   color=color, thickness=3)

                if np.any(cond_plot) : 

                    fig.add_trace(go.Scatter(
                        x=filter_nan(x, cond_plot), y=filter_nan(y, cond_plot),
                        mode='lines',
                        name=date,
                        legendgroup=date,
                        showlegend=date not in dates_in_legend,
                        line=dict(color=color, width=3),
                        error_y=error_y,
                    ))

                    dates_in_legend.add(date)

                    min_wl = min(min_wl, np.nanmin(wl[cond_plot]))
                    max_wl = max(max_wl, np.nanmax(wl[cond_plot]))

                    min_x = min(min_x, np.nanmin(x[cond_plot]))
                    max_x = max(max_x, np.nanmax(x[cond_plot]))


    # Spectre de référence
    if plot_reference is not None:
        spline        = spec_dic['reference'][inst][plot_reference]['spline']
        valid_domains = spec_dic['reference'][inst][plot_reference]['valid_domains']

        x_ref = np.linspace(min_wl, max_wl, int((max_wl - min_wl)/ 0.01))
        y_ref = spline(x_ref)

        cond_plot = np.zeros_like(x_ref, dtype=bool)
        for domain in valid_domains:
            cond_plot |= (x_ref >= domain[0]) & (x_ref <= domain[1])

        if line is not None: x_ref = (x_ref - line) / line * const_c_km

        if xlim is not None: cond_plot &= ((x_ref >= xlim[0]) & (x_ref <= xlim[1]))

        fig.add_trace(go.Scatter(
            x=filter_nan(x_ref, cond_plot), y=filter_nan(y_ref, cond_plot),
            mode='lines',
            name=plot_reference,
            line=dict(color='black', width=4),
        ))



    # Axes et mise en forme
    xlabel = "Velocity (km/s)" if line is not None else "Wavelength (Å)"

    fig.update_layout(
        font=dict(family='STIX Two Text'),
        title=dict(
        font=dict(size=22),
        text=title,
        x=0.5,        # centré
        y=0.95,
        xanchor='center',
       ),
        width=700, height=400,
        xaxis=dict(
            title=dict(text=xlabel, font=dict(size=22), standoff=20),
            range=xlim if xlim is not None else [min_x - 0.05*(max_x-min_x), max_x + 0.05*(max_x-min_x)],
            automargin=True,
            tickfont=dict(size=16),
            ticklabelstandoff=-5

        ),
        yaxis=dict(
            title=dict(text="Flux (erg/s/cm²/Å)", font=dict(size=22)),
            range=ylim,
            rangemode='nonnegative',
            exponentformat='power',   # affiche 2·10^XX  (plus propre)
            showexponent='all',       # applique le format à tous les ticks
            automargin=True,
            tickfont=dict(size=16),
        ),
        legend=dict(font=dict(size=16), borderwidth=1, tracegroupgap=0, **legend_pos),
        margin=dict(l=140, b=80, t=60, autoexpand=False, pad = 10),
        template='plotly_white',
        shapes=[dict(
        type='rect',
        xref='paper', yref='paper',
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color='black', width=1),
        )]
    )


    fig.show()

    total_points = sum(len(t.x) for t in fig.data if t.x is not None)
    print(f"Nombre total de points : {total_points:,}")


    fig_json = fig.to_json()
    size_mb = sys.getsizeof(fig_json) / 1024**2
    print(f"Taille figure : {size_mb:.2f} MB")

    return None  




# Function to plot the modelled spectrum and compare it to observations
#    + The high-resolution model spectrum can be shown with plot_model_HR = True
#    + The model spectrum resampled on the instrumental WL tables can be shown with plot_model_LR = True
#    + The observed data is also shown
#    + By default, the spectra are shown on the full wavelength table
#         - It is possible to plot a single line, using line = [2740.358] (for instance, to plot the Fe II line at 1740.358 A)
#         - In this case, adjust xlim (e.g. [-150,150] km/s)
def plot_exocomet_model(Analysis_dic, dic_flux_all_orders, n_comp, plot_model_HR = True, plot_model_LR = False, plot_error_bar = False,
                    xlim = None, ylim = [0, None], line = None, title = None) : 
    
    fact_redshift = (1 + rv_Beta_Pic/const_c_km)


        
    fig = go.Figure()
    labels_in_legend = set()

    min_x,  max_x  = np.inf, -np.inf
    min_wl,  max_wl  = np.inf, -np.inf

    # Plot observations
    for inst in dic_flux_all_orders : 
        for date in dic_flux_all_orders[inst] : 
            for spec in dic_flux_all_orders[inst][date] : 
                for i_ord in dic_flux_all_orders[inst][date][spec] :

                    # Wavelength tables
                    wl_ord_LR = dic_flux_all_orders[inst][date][spec][i_ord]['wl_ord_LR']

                    # Shift to RV if necessary
                    if line is not None : x_ord_LR = (wl_ord_LR - line)/line * const_c_km
                    else                : x_ord_LR = wl_ord_LR

                    # Conversion from 0-1 scale to calibrated flux (erg/s/cm2/A)
                    #    + For HARPS (not calibrated), we set the mean flux in the 3900-4000 A range to be 1.8 x 10^10 erg/s/cm2/A 
                    func_continuum = Analysis_dic['Continuum'][inst]['spline']
                    if inst == 'HARPS' : 
                        fact_harps = 1.8e-10 / np.nanmean( (dic_flux_all_orders[inst][date][spec][i_ord]['Flux_obs_LR'] * func_continuum(wl_ord_LR * fact_redshift))  [ (wl_ord_LR > 3940) & (wl_ord_LR < 3960) ])
                        func_continuum_plot = lambda wl : fact_harps * func_continuum(wl * fact_redshift)
                    else : func_continuum_plot = lambda wl : func_continuum(wl * fact_redshift)

                    # Flux
                    y = dic_flux_all_orders[inst][date][spec][i_ord]['Flux_obs_LR']  * func_continuum_plot(wl_ord_LR)
                        
                    # Error bars 
                    if plot_error_bar : 
                        error_y=dict(type='data', 
                                     array=dic_flux_all_orders[inst][date][spec][i_ord]['Error_obs_LR'] * func_continuum_plot(wl_ord_LR),
                                     visible=True, color='dimgrey', thickness=3, width=0,
                            )
                    else : error_y = None

                    # Plot
                    if xlim is None or (np.sum((x_ord_LR > xlim[0]) & (x_ord_LR < xlim[1])) > 0) : 

                        fig.add_trace(go.Scatter(
                            x=x_ord_LR, 
                            y=y,
                            error_y=error_y,
                            mode='lines',
                            name=date,
                            legendgroup=date,
                            showlegend=date not in labels_in_legend,
                            line=dict(color='dimgrey', width=3, dash = 'solid'),
                        ))


                        labels_in_legend.add(date)

                        min_x = min(min_x, np.nanmin(x_ord_LR))
                        max_x = max(max_x, np.nanmax(x_ord_LR))

                        min_wl = min(min_wl, np.nanmin(wl_ord_LR))
                        max_wl = max(max_wl, np.nanmax(wl_ord_LR))



    # Plot individual components + ISM
    if plot_model_HR : 
        for inst in dic_flux_all_orders : 
            for date in dic_flux_all_orders[inst] : 
                for spec in dic_flux_all_orders[inst][date] : 
                    for i_ord in dic_flux_all_orders[inst][date][spec] :

                        # Wavelength tables
                        wl_ord_HR =  dic_flux_all_orders[inst][date][spec][i_ord]['wl_ord_HR']

                        # Shift to RV if necessary
                        if line is not None : x_ord_HR = (wl_ord_HR - line)/line * const_c_km
                        else                : x_ord_HR = wl_ord_HR

                        # Conversion from 0-1 scale to calibrated flux (erg/s/cm2/A)
                        #    + For HARPS (not calibrated), we set the mean flux in the 3900-4000 A range to be 1.8 x 10^10 erg/s/cm2/A 
                        func_continuum = Analysis_dic['Continuum'][inst]['spline']
                        if inst == 'HARPS' : 
                            func_continuum_plot = lambda wl : fact_harps * func_continuum(wl * fact_redshift)
                        else : func_continuum_plot = lambda wl : func_continuum(wl * fact_redshift)

                        if xlim is None or (np.sum((x_ord_HR > xlim[0]) & (x_ord_HR < xlim[1])) > 0) : 
                            
                            # ISM
                            flux_ISM_HR = copy.deepcopy(dic_flux_all_orders[inst][date][spec][i_ord]['Flux_ISM_HR'])
                            cond = (flux_ISM_HR < 0.999) 
                            flux_ISM_HR = np.nan
                            
                            if np.any(cond) : 
                                fig.add_trace(go.Scatter(
                                    x=filter_nan(x_ord_HR, cond),
                                    y=filter_nan(flux_ISM_HR * func_continuum_plot(wl_ord_HR), cond),
                                    mode='lines',
                                    name = 'ISM',
                                    showlegend=False,
                                    line=dict(color='grey', width=2.5, dash = 'dashdot'),
                                ))

                                min_x = min(min_x, np.nanmin(x_ord_HR))
                                max_x = max(max_x, np.nanmax(x_ord_HR))
                                min_wl = min(min_wl, np.nanmin(wl_ord_LR))
                                max_wl = max(max_wl, np.nanmax(wl_ord_LR))

                            # Gaseous components
                            for i_comp in range(n_comp) : 
                                if date == '2025-04-29' : 
                                    color_loc = {0 : 'dodgerblue', 1 : 'forestgreen', 2 : '#FF7F0E', 3 : 'black'}[i_comp]
                                    ls_loc = {0 : 'solid', 1 : 'solid', 2 : 'solid', 3 : '5px,3px'}[i_comp]
                                    name_loc = {0 : 'LVC 1', 1 : 'LVC 2', 2 : 'LVC 3', 3 : 'disc'}[i_comp]
                                if date == '2025-09-10' : 
                                    color_loc = {0 : 'cyan', 1 : 'darkviolet', 2 : 'grey', 3 : 'black'}[i_comp]
                                    ls_loc = {0 : 'solid', 1 : 'solid', 2 : 'solid', 3 : '5px,3px'}[i_comp]
                                    name_loc = {0 : 'LVC 5', 1 : 'LVC 4', 2 : 'Faint comet', 3 : 'disc'}[i_comp]
                                color_loc = apply_alpha_on_white(color_loc, 0.5)

                                flux_comp = copy.deepcopy(dic_flux_all_orders[inst][date][spec][i_ord]['Flux_comp_'+str(i_comp)+'_HR'])
                                cond = (flux_comp < 0.999) 
                                if np.any(cond) : 
                                    fig.add_trace(go.Scatter(
                                    x=filter_nan(x_ord_HR, cond),
                                    y=filter_nan(flux_comp * func_continuum_plot(wl_ord_HR), cond),
                                    mode='lines',
                                    name = name_loc,
                                    showlegend=False,
                                    line=dict(color=color_loc, width=2.5, dash = ls_loc),
                                    ))


                                    min_x = min(min_x, np.nanmin(x_ord_HR))
                                    max_x = max(max_x, np.nanmax(x_ord_HR))

                                    min_wl = min(min_wl, np.nanmin(wl_ord_LR))
                                    max_wl = max(max_wl, np.nanmax(wl_ord_LR))

    # Plot model at HR
    if plot_model_HR : 
        for inst in dic_flux_all_orders : 
            for date in dic_flux_all_orders[inst] : 
                for spec in dic_flux_all_orders[inst][date] : 
                    for i_ord in dic_flux_all_orders[inst][date][spec] :

                        # Wavelength tables
                        wl_ord_HR =  dic_flux_all_orders[inst][date][spec][i_ord]['wl_ord_HR']

                        # Shift to RV if necessary
                        if line is not None : x_ord_HR = (wl_ord_HR - line)/line * const_c_km
                        else                : x_ord_HR = wl_ord_HR

                        # Conversion from 0-1 scale to calibrated flux (erg/s/cm2/A)
                        #    + For HARPS (not calibrated), we set the mean flux in the 3900-4000 A range to be 1.8 x 10^10 erg/s/cm2/A 
                        func_continuum = Analysis_dic['Continuum'][inst]['spline']
                        if inst == 'HARPS' : 
                            func_continuum_plot = lambda wl : fact_harps * func_continuum(wl * fact_redshift)
                        else : func_continuum_plot = lambda wl : func_continuum(wl * fact_redshift)

                        if xlim is None or (np.sum((x_ord_HR > xlim[0]) & (x_ord_HR < xlim[1])) > 0) : 

                            flux_model = dic_flux_all_orders[inst][date][spec][i_ord]['Flux_model_HR_convolved']
                            cond = (flux_model < 0.999) 
                            # flux_model[~cond] = np.nan
                            if np.any(cond) :
                                fig.add_trace(go.Scatter(
                                x=filter_nan(x_ord_HR, cond),
                                y=filter_nan(flux_model * func_continuum_plot(wl_ord_HR), cond),
                                mode='lines',
                                name='Exocomet model',
                                legendgroup='Exocomet model',
                                showlegend='Exocomet model' not in labels_in_legend,
                                line=dict(color='red', width=3, dash = 'solid'),
                            ))
                                
                                labels_in_legend.add('Exocomet model')

                                min_x = min(min_x, np.nanmin(x_ord_HR))
                                max_x = max(max_x, np.nanmax(x_ord_HR))

                                min_wl = min(min_wl, np.nanmin(wl_ord_LR))
                                max_wl = max(max_wl, np.nanmax(wl_ord_LR))


    # Plot model at LR
    if plot_model_LR : 
        for inst in dic_flux_all_orders : 
            for date in dic_flux_all_orders[inst] : 
                for spec in dic_flux_all_orders[inst][date] : 
                    for i_ord in dic_flux_all_orders[inst][date][spec] :

                        # Wavelength tables
                        wl_ord_LR = dic_flux_all_orders[inst][date][spec][i_ord]['wl_ord_LR']

                        # Shift to RV if necessary
                        if line is not None : x_ord_LR = (wl_ord_LR - line)/line * const_c_km
                        else                : x_ord_LR = wl_ord_LR
                            
                        # Conversion from 0-1 scale to calibrated flux (erg/s/cm2/A)
                        #    + For HARPS (not calibrated), we set the mean flux in the 3900-4000 A range to be 1.8 x 10^10 erg/s/cm2/A 
                        func_continuum = Analysis_dic['Continuum'][inst]['spline']
                        if inst == 'HARPS' : 
                            func_continuum_plot = lambda wl : fact_harps * func_continuum(wl * fact_redshift)
                        else : func_continuum_plot = lambda wl : func_continuum(wl * fact_redshift)

                        # Plot
                        if xlim is None or (np.sum((x_ord_LR > xlim[0]) & (x_ord_LR < xlim[1])) > 0) : 

                            flux_model = copy.deepcopy(dic_flux_all_orders[inst][date][spec][i_ord]['Flux_model_LR'])
                            cond_fit_order = dic_flux_all_orders[inst][date][spec][i_ord]['cond_fit_LR']
                            flux_model[~cond_fit_order] = np.nan
                            if np.any(cond_fit_order) : 
                                fig.add_trace(go.Scatter(
                                x=wl_ord_LR, 
                                y=flux_model * func_continuum_plot(wl_ord_LR),
                                mode='lines',
                                name='Exocomet model - rebined',
                                legendgroup='Exocomet model - rebined',
                                showlegend='Exocomet model - rebined' not in labels_in_legend,
                                line=dict(color='indigo', width=3, dash = 'solid'),
                                ))

                                labels_in_legend.add('Exocomet model - rebined')

                                min_x = min(min_x, np.nanmin(x_ord_LR))
                                max_x = max(max_x, np.nanmax(x_ord_LR))

                                min_wl = min(min_wl, np.nanmin(wl_ord_LR))
                                max_wl = max(max_wl, np.nanmax(wl_ord_LR))

    # Plot stellar continuum
    for inst in dic_flux_all_orders : 
        func_continuum = Analysis_dic['Continuum'][inst]['spline']
        valid_domains = Analysis_dic['Continuum'][inst]['valid_domains']
        if inst == 'HARPS' : 
            func_continuum_plot = lambda wl : fact_harps * func_continuum(wl * fact_redshift)
        else : func_continuum_plot = lambda wl : func_continuum(wl * fact_redshift)

        # Wavelength table
        wl_continuum = np.linspace(min_wl, max_wl, int((max_wl - min_wl)/0.01)+1)
        
        cond = np.zeros_like(wl_continuum, dtype = bool)
        for domain in valid_domains:
            cond |= (wl_continuum >= domain[0]) & (wl_continuum <= domain[1])

        # Shift to RV if necessary
        if line is not None : x_continuum = (wl_continuum - line)/line * const_c_km
        else                : x_continuum = wl_continuum

        if xlim is not None: cond &= ((x_continuum >= xlim[0]) & (x_continuum <= xlim[1]))




        fig.add_trace(go.Scatter(
            x=filter_nan(x_continuum, cond),
            y=filter_nan(func_continuum_plot(wl_continuum), cond),
            mode='lines',
            name='Reference spectrum',
            legendgroup='Reference spectrum',
            showlegend='Reference spectrum' not in labels_in_legend,
            line=dict(color='black', width=3, dash = 'solid'),
        ))


        labels_in_legend.add('Reference spectrum')

        min_x = min(min_x, np.nanmin(x_ord_LR))
        max_x = max(max_x, np.nanmax(x_ord_LR))
                            
    # Axes et mise en forme
    xlabel = "Velocity (km/s)" if line is not None else "Wavelength (Å)"

    fig.update_layout(
        font=dict(family='STIX Two Text'),
        title=dict(
        text=title,
        font=dict(size=22),
        x=0.5,        # centré
        y=0.96,
        xanchor='center',
       ),
        width=700, height=400,
        xaxis=dict(
            title=dict(text=xlabel, font=dict(size=20), standoff=20),
            range=xlim if xlim is not None else [min_x - 0.05*(max_x-min_x), max_x + 0.05*(max_x-min_x)],
            automargin=False,
            tickfont=dict(size=16),
            ticklabelstandoff=-5
        ),
        yaxis=dict(
            title=dict(text="Flux (erg/s/cm²/Å)", font=dict(size=20)),
            range=ylim,
            rangemode='nonnegative',
            exponentformat='power',   # affiche 2·10^XX  (plus propre)
            showexponent='all',       # applique le format à tous les ticks
            automargin=False,
            tickfont=dict(size=16),
        ),
        legend=dict(font=dict(size=16), x=0.01, y=0.99, xanchor='left',   yanchor='top', borderwidth=1, tracegroupgap=0),
        margin=dict(l=140, b=80, t=60, autoexpand=False, pad = 10),
        template='plotly_white',
        shapes=[dict(
        type='rect',
        xref='paper', yref='paper',
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color='black', width=1),
        )]
    )
    


    fig.show()

    total_points = sum(len(t.x) for t in fig.data if t.x is not None)
    print(f"Nombre total de points : {total_points:,}")


    fig_json = fig.to_json()
    size_mb = sys.getsizeof(fig_json) / 1024**2
    print(f"Taille figure : {size_mb:.2f} MB")
    

    return None




