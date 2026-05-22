# Exocomet spectra - Beta Pictoris
This code models the absorption signatures of transiting exocomet gas tails in atomic lines.

## Overview

The absorption spectrum of a comet is computed from a small set of physical parameters, such as the column densities of modelled species (e.g. Fe II, C I) and the comet's distance to the star. The excitation state of the gaseous tail is derived from radiative and collisional excitation, accounting for self-shielding of stellar photons. The resulting model can then be compared to observations to retrieve the physical and chemical properties of transiting exocomets detected in spectroscopy.

An example application is provided in Plots_spectra.ipynb. The notebook lets you:
 
- Explore spectroscopic observations of Beta Pictoris obtained with the Hubble Space Telescope on April 29 and September 10, 2025;
- Visualise the detected exocomet signatures interactively;
- Run the exocomet model (normally takes a few seconds on a cpu);
- Compare exocomet observations against the best-fit solution.

The rendered notebook is available at : https://tvrignaud.github.io/Exocomet_spectrum

## Repository structure
- `Plots_spectra.qmd` : Quarto notebook
- `Plots_spectra.ipynb` : Jupyter notebook
- `Routines.py` : Core functions (Model computation, plot functions)
- `List_studied_lines.py` : List of fitted spectral lines
- `Settings_2025_04_29_refrac_carbon.py` : Custom settings for fitting the April 29, 2025 observations
- `Settings_2025_09_10_refrac_carbon.py` : Custom settings for fitting the September 10, 2025 observations

## Data
The data files required to run this notebook are hosted on Zenodo : https://doi.org/10.5281/zenodo.19072081 : 
- `Data_Beta_Pic.npy` : spectroscopic observations of Beta Pic obtained with the Hubble Space Telescope and the HARPS spectrograph
- `Data_tabulated.zip` : tabulated spectroscopic data for S I, Ca II, Mn II, Fe II, Si II, Cr II, Ni II, and C I. Also includes a spectral model of Beta Pictoris, and tabulated line spread functions for the STIS and COS instruments.

Download `Data_Beta_Pic.npy` and `Data_tabulated.zip`, unzip the latter, and place them in the root folder.

## Dependencies
- Python 3.12.1
- `numpy`, `scipy`, `pandas`, `astropy`, `matplotlib`, `plotly`, `ipykernel`
- `bindensity` (Bourrier et al. 2025, A&A 691, A113; https://www.astro.unige.ch/~delisle/bindensity/doc/)

## Code use
After cloning the GitHub repository and installing the dependencies -- which should take only a few minutes -- the code use is detailed in the notebook `Plots_spectra.ipynb`.

## References
- Vrignaud & Lecavelier (2026, A&A, 707, A60; https://www.aanda.org/articles/aa/abs/2026/03/aa57819-25)
- Vrignaud et al. (in prep., 2026; https://www.researchsquare.com/article/rs-9515646/v1)

## Contact
Théo Vrignaud — vrignaud@iap.fr

## License
This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt this material for any purpose, provided you give appropriate credit.