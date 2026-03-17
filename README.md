# Exocomet spectra — Beta Pictoris
Interactive visualisation of exocomet absorption signatures detected in Beta Pictoris spectra obtained on April 29 and September 10, 2025.

## Interactive notebook
The rendered notebook is available at : https://tvrignaud.github.io/Exocomet_spectrum

## Repository structure
- `Plots_spectra.qmd` — Quarto notebook (source)
- `Plots_spectra.ipynb` — Jupyter notebook (source)
- `Routines.py` — Core functions (Model computation, plot functions)
- `List_studied_lines.py` — List of fitted spectral lines
- `Settings_2025_04_29_refrac_carbon.py` — Custom settings for fitting the April 29, 2025 observations (3 exocomets)
- `Settings_2025_09_10_refrac_carbon.py` — Custom settings for fitting the September 10, 2025 observations (3 exocomets)

## Data
The data files required to run this notebook are hosted on Zenodo : https://doi.org/10.5281/zenodo.19072081
Download `Data_Beta_Pic.npy` and `Data_tabulated.zip` and place them in the root folder.

## Dependencies
- Python 3.x
- `numpy`, `scipy`, `plotly`, `astropy`, 
- `bindensity` (Bourrier et al. 2025, A&A 691, A113)

## References
- Vrignaud & Lecavelier (2026, A&A, 707, A60)
- Vrignaud et al. (in prep., 2026)

## Contact
Théo Vrignaud — vrignaud@iap.fr