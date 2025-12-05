#%% Imports
import sys
sys.path.append('../src')

from module_WRA_EYA import load_ncs, get_ws_wd_for_xyz, \
    wbl_fit, plot_wind_rose, get_gross_aep, analysis_case

#%% inputs
# if __name__ == '__main__':
#%% Inputs
# Wind Resource
input_nc_folder = '../inputs'
lat = 55.625 # 55.5° - 55.75°
lon = 7.875 # 7.75° - 8.0°
start_date='1998-01-01'
end_date='2005-12-31'
# WTG
power_curve = '../inputs/NREL_Reference_5MW_126.csv'
hh = 120 # m


# %% Analysis

# Read Project's netcdf
ds = load_ncs(input_nc_folder, start_date, end_date)

# Get ws and wd at coordinates and hub height of interest
ws_interp, wd_interp = get_ws_wd_for_xyz(ds=ds,
                                    hh=hh,
                                    lat=lat,
                                    lon=lon,
                                    exponent=2)

# Get weibull parameters, and plot ws and wd distributions
k, A, fig_ws = wbl_fit(ws_interp, plot=True)
print(f'Weibull parameters: k = {k:,.2f}, and A = {A:,.2f} m/s.')

fig_wd = plot_wind_rose(wd_interp, ws_interp, h=hh)

# Estimate Gross AEP
gross_aep = get_gross_aep(power_curve, ws_interp)
print(f'Gross AEP for coordinates: {lat}°, {lon}° at hub height {hh} m is {gross_aep:,.1f} GWh')

#%% Create an analysis case
case = analysis_case(
    power_curve_path=power_curve,
    input_nc_folder=input_nc_folder,
    start_date=start_date,
    end_date=end_date
)

# Run the calculate method
case.calculate(lat=lat, lon=lon, exponent=2, hubheight=hh)

# Print the analysis case
print(case)