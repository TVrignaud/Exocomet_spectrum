import numpy as np

list_2025_09_10 = ['ofgs05010_x1d.fits', 'ofgs05020_x1d.fits', 'ofgs05030_x1d.fits', 'ofgs05040_x1d.fits', 'ofgs05050_x1d.fits', 'ofgs05060_x1d.fits']





'''

Model settings

'''

settings = {}


####################
### Observations ###
####################

# Path to observations
settings['path_data'] = 'Data_Beta_Pic.npy'

# List of spectra included in the study, per instrument
settings['list_spectra'] = {'STIS' : {'2025-09-10' : list_2025_09_10}, 'HARPS' : {'2025-09-10' : ['Average_night']}, 'COS' : {'2025-09-10' : ['lfgs06010_x1dsum.fits']} }

# Continuum used to extract the absorption spectrum, per instrument
settings['continuum'] = {'STIS' : 'Continuum 2025-09-10 STIS', 
                         'HARPS' : 'Continuum 2025-09-10 HARPS', 
                         'COS' : 'Continuum 2025-09-10 COS'}


# Analyze only a fraction of a spectrum
settings['truncate_data'] = {'HARPS' : [3915,3985], 'COS' : [1200,1420]}

# Increase the level of zero-flux
settings['increase_zero_flux_level'] = {'HARPS' : [([3900,4000], 0.0008)]}

# Line spread functions
settings['LSF_path'] = 'Data_tabulated//LSF'
settings['LSF'] = {'HARPS' : {}, 'STIS' : {}, 'COS' : {}}
settings['LSF']['STIS']['2025-09-10']  = {'ofgs05010_x1d.fits' : 'E230H_1700_0_01x0_03',
                                          'ofgs05020_x1d.fits' : 'E230H_2400_0_1x0_09',
                                          'ofgs05030_x1d.fits' : 'E140H_1500_0_2x0_2',
                                          'ofgs05040_x1d.fits' : 'E230H_1700_0_1x0_2',
                                          'ofgs05050_x1d.fits' : 'E230H_2400_0_01x0_03',
                                          'ofgs05060_x1d.fits' : 'E230H_2400_0_01x0_03'}                    

settings['LSF']['HARPS']['2025-09-10']  = {'Average_night' : 'Brandeker_2011'}                    

settings['LSF']['COS']['2025-09-10'] = {'lfgs06010_x1dsum.fits' : 'COS_FP3_FP4'}





##########################################################
### Espèces chimiques étudiées et spectre de référence ###
##########################################################

# Studied species
settings['list_species'] = ['Fe II', 'Ni II', 'Ca II', 'Cr II', 'Mn II', 'Si II', 'S I',
                            'C I 3P 1/2',   'C I 3P 3/2',   'C I 3P 5/2',   'C I 1D 5/2',
                            '13C I 3P 1/2', '13C I 3P 3/2', '13C I 3P 5/2', '13C I 1D 5/2',
                            ]

# Path towards tabulated spectroscopic data
settings['path_species'] = {'Fe II' : 'Data_tabulated//data_fe2.npy',
                            'Ni II' : 'Data_tabulated//data_ni2.npy',
                            'Ca II' : 'Data_tabulated//data_ca2.npy',
                            'Cr II' : 'Data_tabulated//data_cr2.npy',
                            'Mn II' : 'Data_tabulated//data_mn2.npy',
                            'Si II' : 'Data_tabulated//data_si2.npy',
                            'S I'   : 'Data_tabulated//data_s1.npy',
                            }

# Number of modelled energy levels
settings['n_level_modeled'] = {'Fe II' : 150, 'Ni II' : 70, 'Ca II' : 19, 'Cr II' : 60, 'Mn II' : 70, 'Si II' : 15, 'S I' : 39, 
                               'C I 3P 1/2' : 1,   'C I 3P 3/2' : 1,   'C I 3P 5/2' : 1,   'C I 1D 5/2' : 1,
                               '13C I 3P 1/2' : 1, '13C I 3P 3/2' : 1, '13C I 3P 5/2' : 1, '13C I 1D 5/2' : 1,}


# ISM    
#    + Column density and b : measured for Fe II
#    + For other species, b adjusted assuming T = 7000 K and xi = 1.5 km/s (Strom 2017)
#    + Depletion taken into account for Si, Zn and Cu (Jenkins 2009)
N_Fe2_ISM = 0.12
v_ISM = -10
b_fe2 = 2.2
xi = 1.6
T_ISM = 7000
settings['ISM'] = {'Fe II' : {'N' : N_Fe2_ISM,                     'b' : np.sqrt(xi**2 + 0.016514*T_ISM/56), 'v' : v_ISM},
                   'Ni II' : {'N' : N_Fe2_ISM*0.056,               'b' : np.sqrt(xi**2 + 0.016514*T_ISM/58), 'v' : v_ISM},
                   'Ca II' : {'N' : N_Fe2_ISM*0.069*0.014,         'b' : np.sqrt(xi**2 + 0.016514*T_ISM/40), 'v' : v_ISM},
                   'Cr II' : {'N' : N_Fe2_ISM*0.013,               'b' : np.sqrt(xi**2 + 0.016514*T_ISM/52), 'v' : v_ISM},
                   'Mn II' : {'N' : N_Fe2_ISM*0.0085,              'b' : np.sqrt(xi**2 + 0.016514*T_ISM/55), 'v' : v_ISM},
                   'Si II' : {'N' : N_Fe2_ISM*1.02*10**0.5,        'b' : np.sqrt(xi**2 + 0.016514*T_ISM/28), 'v' : v_ISM},
                   }

# Spectre stellaire utilisé
settings['path_reference_spectrum'] = 'Data_tabulated//Model_spectrum_Beta_Pic.npy'

# wl range where the transmission model is computed
settings['wl_range_global'] = [900,9000]

# RV resolution of the fine grid used to compute the transmission model (in km/s)
settings['res_rv_HR'] = 0.3

# RV window around each line where the model is computed 
#    + Must be symetric (for now)
settings['RV_window_model'] = [-50,50]





##########################################
### Select the fitted spectral regions ###
##########################################

# Fitted RV range (km/s)
settings['fitted_rv_range'] = {'STIS' : [-23,42], 'HARPS' : [-20,35], 'COS' : [-25,40]}

# Remove wl ranges from the fit
settings['excluded_ranges'] = {'STIS' : [[2392.06, 2392.17], [2403.25, 2403.31], [2592.56, 2592.62],  [1533.376, 1533.379],   # spurious features
                                         [2749,2751],                                                                         # Avoid the Fe II 2750 A triplet
                                         ]}

# Specify the wl ranges that should be fitted
settings['fitted_ranges'] = {'COS' : [ [1270,1272], [1310,1318], [1328,1331], [1341,1343], [1355,1357], [1369,1371] ] }


# Add systematic uncertainties on each flux pixel
#    + The added error depends on the absorption depth of the pixel
settings['sigma_model'] = [{'inst' : 'STIS', 'plage' : [1500,3000],  'coeff' : [-0.6180234,   1.19197712, -0.80578972,  0.23298165,  0.00552586], 'deg' : 4},
                           {'inst' : 'HARPS', 'plage' : [3900,4000], 'coeff' : [-0.6180234,   1.19197712, -0.80578972,  0.23298165,  0.00552586], 'deg' : 4},
                           ]



#########################################
### Apply RV shifts to specific lines ###
#########################################

# + This corrects for the imperfect calibration of STIS and COS, precise within 1 km/S for STIS and 5 km/s for COS.

dic_shift_stis = {'2025-09-10' : {}}

for name in list_2025_09_10 : 
    dic_shift_stis['2025-09-10'][name] = {}
    for i in range(100) : dic_shift_stis['2025-09-10'][name][i] = []

dic_shift_stis['2025-09-10']['ofgs05030_x1d.fits'][10].append( ([1618,1619], -0.5) )       # 1618 Fe II

dic_shift_stis['2025-09-10']['ofgs05040_x1d.fits'][46].append( ([1701.9,1702.3], 0.7) )    # 1702 Fe II
dic_shift_stis['2025-09-10']['ofgs05040_x1d.fits'][44].append( ([1709,1710], 0.2) )        # 1709 Ni II
dic_shift_stis['2025-09-10']['ofgs05040_x1d.fits'][43].append( ([1709,1710], -0.6) )       # 1709 Ni II
dic_shift_stis['2025-09-10']['ofgs05040_x1d.fits'][35].append( ([1741,1742], -0.7) )       # 1741 Ni II
dic_shift_stis['2025-09-10']['ofgs05040_x1d.fits'][34].append( ([1748,1749], 1.5) )        # 1748 Ni II
dic_shift_stis['2025-09-10']['ofgs05040_x1d.fits'][33].append( ([1748,1748], 1.5) )        # 1748 Ni II
dic_shift_stis['2025-09-10']['ofgs05040_x1d.fits'][32].append( ([1751.8,1752.2], -0.7) )   # 1751 Ni II

dic_shift_stis['2025-09-10']['ofgs05040_x1d.fits'][19].append( ([1807,1897.6], -0.7) )     # 1807 S I

dic_shift_stis['2025-09-10']['ofgs05010_x1d.fits'][14].append( ([2056,2056.5], -1) )       # 2056 Cr II
dic_shift_stis['2025-09-10']['ofgs05010_x1d.fits'][13].append( ([2062,2062.5], -0.8) )     # 2056 Cr II

dic_shift_stis['2025-09-10']['ofgs05050_x1d.fits'][35].append( ([2166,2166.5], 0.5) )      # 2166 Ni II
dic_shift_stis['2025-09-10']['ofgs05050_x1d.fits'][34].append( ([2166,2166.5], -0.2) )     # 2166 Ni II
dic_shift_stis['2025-09-10']['ofgs05050_x1d.fits'][25].append( ([2225,2226], 1) )          # 2225 Ni II
dic_shift_stis['2025-09-10']['ofgs05050_x1d.fits'][21].append( ([2249.5,2250.3], -0.8) )   # 2249 Fe II
dic_shift_stis['2025-09-10']['ofgs05050_x1d.fits'][20].append( ([2260.5,2261], -0.4) )     # 2260 Fe II
dic_shift_stis['2025-09-10']['ofgs05050_x1d.fits'][19].append( ([2260.5,2261], -0.8) )     # 2260 Fe II

dic_shift_stis['2025-09-10']['ofgs05050_x1d.fits'][12].append( ([2316.5,2317],  0.3) )     # 2316 Ni II
dic_shift_stis['2025-09-10']['ofgs05050_x1d.fits'][11].append( ([2316.5,2317], -0.4) )     # 2316 Ni II

dic_shift_stis['2025-09-10']['ofgs05050_x1d.fits'][8].append( ([2338.5,2339.5], -0.6) )    # 2338 Fe II
dic_shift_stis['2025-09-10']['ofgs05050_x1d.fits'][7].append( ([2344,2344.5], -0.4) )      # 2344 Fe II
dic_shift_stis['2025-09-10']['ofgs05050_x1d.fits'][5].append( ([2362.4,2363.2], 0.7) )     # 2362 Fe II
dic_shift_stis['2025-09-10']['ofgs05050_x1d.fits'][3].append( ([2374,2375], -0.4) )        # 2374 Fe II

dic_shift_stis['2025-09-10']['ofgs05060_x1d.fits'][32].append( ([2381,2383.5], -0.5) )     # 2382 Fe II
dic_shift_stis['2025-09-10']['ofgs05060_x1d.fits'][31].append( ([2389,2390], -0.5) )       # 2389 Fe II
dic_shift_stis['2025-09-10']['ofgs05060_x1d.fits'][30].append( ([2396,2397], -0.5) )       # 2396 Fe II
dic_shift_stis['2025-09-10']['ofgs05060_x1d.fits'][28].append( ([2411,2412], -0.4) )       # 2411 Fe II
dic_shift_stis['2025-09-10']['ofgs05060_x1d.fits'][17].append( ([2499.5,2500], 1) )        # 2499 Fe II
dic_shift_stis['2025-09-10']['ofgs05060_x1d.fits'][6].append( ([2592,2592.8], 0.5) )       # 2592 Fe II
dic_shift_stis['2025-09-10']['ofgs05060_x1d.fits'][3].append( ([2621,2622], 1) )           # 2621 Fe II



dic_shift_cos = {'2025-09-10' : {}}

for name in ['lfgs06010_x1dsum.fits'] : 
    dic_shift_cos['2025-09-10'][name] = {}
    for i in range(100) : dic_shift_cos['2025-09-10'][name][i] = []

dic_shift_cos['2025-09-10']['lfgs06010_x1dsum.fits'][1].append( ([1250,1260], -3) )        # 1250 S II

dic_shift_cos['2025-09-10']['lfgs06010_x1dsum.fits'][0].append( ([1317,1317.4], -3) )      # 1317 Ni II
dic_shift_cos['2025-09-10']['lfgs06010_x1dsum.fits'][0].append( ([1369,1371], -3) )        # 1370 Ni II

dic_shift_cos['2025-09-10']['lfgs06010_x1dsum.fits'][0].append( ([1328,1330], -3) )        # 1330 C I
dic_shift_cos['2025-09-10']['lfgs06010_x1dsum.fits'][0].append( ([1355,1357], -8) )        # 1355 C I

settings['RV_shift_lines'] = {
    'STIS' : dic_shift_stis,                                                             
    'COS'  : dic_shift_cos                                                                   
    }


