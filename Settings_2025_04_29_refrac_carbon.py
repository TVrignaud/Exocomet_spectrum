import numpy as np

list_2025_04_29 = ['ofgs01010_x1d.fits', 'ofgs01020_x1d.fits', 'ofgs01030_x1d.fits', 'ofgs01040_x1d.fits', 'ofgs01050_x1d.fits', 'ofgs01060_x1d.fits']






'''

Model settings

'''

settings = {}


####################
### Observations ###
####################

# Path to observations
settings['path_data'] = 'Data_Beta_Pic.npy'

# List of spectra to include in the tudy, per instrument
settings['list_spectra'] = {'STIS' : {'2025-04-29' : list_2025_04_29}, 'HARPS' : {'2025-04-29' : ['Average_night']}, 'COS' : {'2025-04-29' : ['lfgs02010_x1dsum.fits']} }

# Continuum used to extract the absorption spectrum, per instrument
settings['continuum'] = {'STIS' : 'Continuum 2025-04-29 STIS', 
                         'HARPS' : 'Continuum 2025-04-29 HARPS', 
                         'COS' : 'Continuum 2025-04-29 COS'}

# Analyze only a fraction of a spectrum
settings['truncate_data'] = {'HARPS' : [3915,3985], 'COS' : [1200,1420]}

# Increase the level of zero-flux
settings['increase_zero_flux_level'] = {'HARPS' : [([3900,4000], 0.0008)]}           

# Line spread functions
settings['LSF_path'] = 'Data_tabulated//LSF'
settings['LSF'] = {'HARPS' : {}, 'STIS' : {}, 'COS' : {}}
settings['LSF']['STIS']['2025-04-29']  = {'ofgs01010_x1d.fits' : 'E230H_1700_0_01x0_03',
                                          'ofgs01020_x1d.fits' : 'E230H_2400_0_1x0_09',
                                          'ofgs01030_x1d.fits' : 'E140H_1500_0_2x0_2',
                                          'ofgs01040_x1d.fits' : 'E230H_1700_0_1x0_2',
                                          'ofgs01050_x1d.fits' : 'E230H_2400_0_01x0_03',
                                          'ofgs01060_x1d.fits' : 'E230H_2400_0_01x0_03'}                    

settings['LSF']['HARPS']['2025-04-29']  = {'Average_night' : 'Brandeker_2011'}                    

settings['LSF']['COS']['2025-04-29'] = {'lfgs02010_x1dsum.fits' : 'COS_FP3_FP4'}



##############################################
### Studied species and reference spectrum ###
##############################################

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

# Model Beta Spectrum
settings['path_reference_spectrum'] = 'Data_tabulated//Model_spectrum_Beta_Pic.npy'

# wl range where the transmission model is computed
settings['wl_range_global'] = [900,9000]

# RV resolution of the fine grid used to compute the transmission model (in km/s)
settings['res_rv_HR'] = 0.3

# RV window around each line where the model is computed 
#    + Must be symetric (for now)
settings['RV_window_model'] = [-30,30]




##########################################
### Select the fitted spectral regions ###
##########################################

# Fitted RV range (km/s)
settings['fitted_rv_range'] = {'STIS' : [-20,18], 'HARPS' : [-20,18], 'COS' : [-25,20]}

# Remove wl ranges from the fit
settings['excluded_ranges'] = {'STIS' : [ [1574.6,1575.2], [2403.15,2403.25],  [2672,2673],  [2750.1, 2751]]}


# Specify the wl ranges that should be fitted
settings['fitted_ranges'] = {'COS' : [ [1270,1272], [1310,1318], [1328,1331], [1341,1343], [1355,1357], [1369,1371] ] }

# Add systematic uncertainties on each flux pixel
#    + The added error depends on the absorption depth of the pixel
settings['sigma_model'] = [{'inst' : 'STIS',  'plage' : [1500,3000], 'coeff' : [-0.36730066,  0.7386761,  -0.53012492,  0.16526333,  0.00751395], 'deg' : 4},
                           {'inst' : 'HARPS', 'plage' : [3900,4000], 'coeff' : [-0.36730066,  0.7386761,  -0.53012492,  0.16526333,  0.00751395], 'deg' : 4},
                           ]


#########################################
### Apply RV shifts to specific lines ###
#########################################

# + This corrects for the imperfect calibration of STIS and COS, precise within 1 km/S for STIS and 5 km/s for COS.

dic_shift_stis = {'2025-04-29' : {}}

for name in list_2025_04_29 : 
    dic_shift_stis['2025-04-29'][name] = {}
    for i in range(100) : dic_shift_stis['2025-04-29'][name][i] = []

# Fe II
dic_shift_stis['2025-04-29']['ofgs01030_x1d.fits'][10].append( ([1618,1619], -0.5) )         # 1618 Fe II
dic_shift_stis['2025-04-29']['ofgs01030_x1d.fits'][8].append( ([1628.5,1629.5], -0.8) )      # 1629 Fe II
dic_shift_stis['2025-04-29']['ofgs01030_x1d.fits'][7].append( ([1635,1638], -0.5) )          # 1636 Fe II

dic_shift_stis['2025-04-29']['ofgs01040_x1d.fits'][46].append( ([1701.5,1702.5], 0.7) )      # 1702 Fe II
dic_shift_stis['2025-04-29']['ofgs01040_x1d.fits'][41].append( ([1720.1,1702.1], 0.5) )      # 1720 Fe II

dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][21].append( ([2249.4,2250.4],   -0.8) )   # 2249 Fe II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][19].append( ([2260.25,2261.25], -0.8) )   # 2260 Fe II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][17].append( ([2280.1,2281.1], 0.5) )      # 2280 Fe II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][16].append( ([2280.1,2281.1], -0.7) )     # 2280 Fe II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][9].append( ([2331.5,2332.5], -0.25) )     # 2332 Fe II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][8].append( ([2338.3,2339.3], -1.) )       # 2338 Fe II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][7].append( ([2343.8,2345.5], -0.5) )      # 2344 Fe II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][5].append( ([2359.3,2360.1], -0.3) )      # 2360 Fe II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][5].append( ([2350.1,2361.5], -0.4) )      # 2360 Fe II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][5].append( ([2365,2366], 0.5) )           # 2365 Fe II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][3].append( ([2374,2375], -0.5) )          # 2374 Fe II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][1].append( ([2389,2390], -0.3) )          # 2389 Fe II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][0].append( ([2399.5,2400.5], 0.5) )       # 2400 Fe II

dic_shift_stis['2025-04-29']['ofgs01060_x1d.fits'][32].append( ([2381,2383.2], -0.5) )       # 2381-2382 Fe II
dic_shift_stis['2025-04-29']['ofgs01060_x1d.fits'][31].append( ([2389,2390], -0.6) )         # 2381-2382 Fe II
dic_shift_stis['2025-04-29']['ofgs01060_x1d.fits'][28].append( ([2410.8,2411.3], -0.2) )     # 2411 Fe II
dic_shift_stis['2025-04-29']['ofgs01060_x1d.fits'][18].append( ([2493.5,2494.5], 0.3) )      # 2494 Fe II
dic_shift_stis['2025-04-29']['ofgs01060_x1d.fits'][17].append( ([2499.1,2500], 0.3) )        # 2499 Fe II
dic_shift_stis['2025-04-29']['ofgs01060_x1d.fits'][6].append( ([2591.8,2592.8], 0.9) )       # 2592 Fe II
dic_shift_stis['2025-04-29']['ofgs01060_x1d.fits'][3].append( ([2612.2,2613.3], 0.9) )       # 2612 Fe II
dic_shift_stis['2025-04-29']['ofgs01060_x1d.fits'][2].append( ([2622,2623], 0.4) )           # 2622 Fe II
dic_shift_stis['2025-04-29']['ofgs01060_x1d.fits'][1].append( ([2631,2633], 0.2) )           # 2632 Fe II

dic_shift_stis['2025-04-29']['ofgs01020_x1d.fits'][20].append( ([2714.8,2715.6], 0.3) )      # 2715 Fe II
dic_shift_stis['2025-04-29']['ofgs01020_x1d.fits'][19].append( ([2727.8,2728.8], 0.7) )      # 2728 Fe II
dic_shift_stis['2025-04-29']['ofgs01020_x1d.fits'][19].append( ([2731,2732], 1.2) )          # 2731 Fe II
dic_shift_stis['2025-04-29']['ofgs01020_x1d.fits'][18].append( ([2737.3,2738.2], 0.4) )      # 2737 Fe II
dic_shift_stis['2025-04-29']['ofgs01020_x1d.fits'][18].append( ([2739.8,2740.9], 0.6) )      # 2740 Fe II
dic_shift_stis['2025-04-29']['ofgs01020_x1d.fits'][17].append( ([2746.8,2748.3], 0.7) )      # 2747 Fe II
dic_shift_stis['2025-04-29']['ofgs01020_x1d.fits'][17].append( ([2749.3,2750.7], 0.5) )      # 2750 Fe II
dic_shift_stis['2025-04-29']['ofgs01020_x1d.fits'][16].append( ([2756,2757], 0.4) )          # 2756 Fe II


# Ni II
dic_shift_stis['2025-04-29']['ofgs01040_x1d.fits'][44].append( ([1709.1,1710.1], 0.5) )      # 1709 Ni II
dic_shift_stis['2025-04-29']['ofgs01040_x1d.fits'][43].append( ([1709.1,1710.1], -0.3) )     # 1709 Ni II
dic_shift_stis['2025-04-29']['ofgs01040_x1d.fits'][36].append( ([1741,1751.2], 0.7) )        # 1741 Ni II
dic_shift_stis['2025-04-29']['ofgs01040_x1d.fits'][34].append( ([1747.8,1748.8], 2.) )       # 1748 Ni II
dic_shift_stis['2025-04-29']['ofgs01040_x1d.fits'][33].append( ([1747.8,1748.8], 2) )        # 1748 Ni II
dic_shift_stis['2025-04-29']['ofgs01040_x1d.fits'][33].append( ([1751.4,1752.4], 0.8) )      # 1751 Ni II
dic_shift_stis['2025-04-29']['ofgs01040_x1d.fits'][20].append( ([1804,1805], 1) )            # 1804 Ni II

dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][35].append( ([2165.7,2166.7], 0.5) )      # 2166 Ni II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][34].append( ([2169.3,2170.3], 1.1) )      # 2169 Ni II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][26].append( ([2216.8,2217.8], 0.2) )      # 2217 Ni II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][25].append( ([2223.1,2224.2], 0.4) )      # 2223 Ni II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][25].append( ([2225,2226.1], 1.2) )        # 2225 Ni II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][21].append( ([2254,2255.1], 0.5) )        # 2254 Ni II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][18].append( ([2270.4,2271.4], 0.6) )      # 2270 Ni II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][16].append( ([2287.3,2288.3], 0.8) )      # 2287 Ni II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][15].append( ([2287.3,2288.3], 0.8) )      # 2287 Ni II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][14].append( ([2297.3,2298.4], 0.8) )      # 2297 Ni II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][13].append( ([2303.2,2304.2], 0.3) )      # 2303 Ni II
dic_shift_stis['2025-04-29']['ofgs01050_x1d.fits'][12].append( ([2316.2,2317.3], 0.8) )      # 2316 Ni II

dic_shift_stis['2025-04-29']['ofgs01060_x1d.fits'][28].append( ([2416.3,2417.4], 0.6) )      # 2416 Ni II
dic_shift_stis['2025-04-29']['ofgs01060_x1d.fits'][27].append( ([2416.3,2417.4], 0.6) )      # 2416 Ni II
dic_shift_stis['2025-04-29']['ofgs01060_x1d.fits'][16].append( ([2511.1,2512.1], 1) )        # 2511 Ni II
dic_shift_stis['2025-04-29']['ofgs01060_x1d.fits'][15].append( ([2511.1,2512.1], 1) )        # 2511 Ni II


# Cr II, Zn II
dic_shift_stis['2025-04-29']['ofgs01010_x1d.fits'][20].append( ([2025,2027], -0.6) )         # 2026 Zn II
dic_shift_stis['2025-04-29']['ofgs01010_x1d.fits'][14].append( ([2055.7,2056.8], -1) )       # 2056 Cr II
dic_shift_stis['2025-04-29']['ofgs01010_x1d.fits'][13].append( ([2061.7,2063.3], -1) )       # 2062 Cr II + Zn II

dic_shift_stis['2025-04-29']['ofgs01020_x1d.fits'][24].append( ([2677.5, 2678.5], -0.5) )    # 2677 Cr II
dic_shift_stis['2025-04-29']['ofgs01020_x1d.fits'][8].append(  ([2836,2837], 0.3) )          # 2836 Cr II
dic_shift_stis['2025-04-29']['ofgs01020_x1d.fits'][8].append(  ([2843.5,2844.6], 0.5) )      # 2844 Cr II
dic_shift_stis['2025-04-29']['ofgs01020_x1d.fits'][7].append(  ([2843.5,2844.6], 0.5) )      # 2844 Cr II
dic_shift_stis['2025-04-29']['ofgs01020_x1d.fits'][7].append(  ([2850.2,2851.2], 0.5) )      # 2844 Cr II


# Si II
dic_shift_stis['2025-04-29']['ofgs01040_x1d.fits'][19].append( ([1807.6,1808.5], 0.6) )      # 1808 Si II
dic_shift_stis['2025-04-29']['ofgs01040_x1d.fits'][17].append( ([1816.4,1817.4], 0.3) )      # 1816 Si II





dic_shift_cos = {'2025-04-29' : {}}

for name in ['lfgs02010_x1dsum.fits'] : 
    dic_shift_cos['2025-04-29'][name] = {}
    for i in range(100) : dic_shift_cos['2025-04-29'][name][i] = []

# C I 
dic_shift_cos['2025-04-29']['lfgs02010_x1dsum.fits'][0].append( ([1310.5,1312], 10) )         # 1311
dic_shift_cos['2025-04-29']['lfgs02010_x1dsum.fits'][0].append( ([1355,1357], -10) )          # 1355



settings['RV_shift_lines'] = {
    'STIS' : dic_shift_stis,                                                             
    'COS'  : dic_shift_cos                                                                   
    }
