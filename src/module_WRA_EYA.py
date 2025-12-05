#%% Module's imports
import xarray as xr
from glob import glob
import numpy as np
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from scipy.interpolate import interp1d
#import pandas as pd

#%% Module's Custom Functions
def load_ncs(input_nc_folder = '../inputs',
            start_date='1997-01-01',
            end_date='2008-12-31'):
    """
    Open multiple NetCDF files from a folder, combine them into a single
    time-sorted dataset, and slice the dataset based on optional dates.
    """
    # Get a list of all .nc files
    nc_files = glob(f"{input_nc_folder}/*.nc")

    # Open the files as a single dataset using MFDataset
    ds = xr.open_mfdataset(nc_files, combine='by_coords')

    # Sort ds by time
    ds_sorted = ds.sortby('time')

    # Slice ds based on the start and end dates
    ds_sliced = ds_sorted.sel(time=slice(start_date, end_date))

    return ds_sliced


def get_weights(ds, lat_target, lon_target, exponent):
    """
    Get weights for 2D interpolation depending on distance to grid-points.
    """
    # Calculate the distances to the target lat and lon
    lat_delta = ds.latitude - lat_target
    lon_delta = ds.longitude - lon_target
    distances = (lat_delta ** 2 + lon_delta **2) ** 0.5
    
    # Avoid division by zero
    distances = distances.where(distances > 0, 1e-10)

    # Calculate weighting
    weights = 1 / (distances ** exponent)

    # Normalize weighting
    weights = weights / weights.sum()

    return weights


def get_ws(u, v):
    """
    Get ws as f(u, v).
    """
    # Derive wind speed
    ws = np.sqrt(u**2 + v**2)

    return ws


def get_wd(u, v):
    """
    Get wd as f(u, v).
    """
    # Derive wind direction
    wd = (270 - np.degrees(np.arctan2(v, u))) % 360

    return wd


def get_uv_interpolated_ws_wd(ds, lat_target, lon_target, exponent, h):
    # Get the weights for the 2D interpolation
    weights = get_weights(ds, lat_target, lon_target, exponent)

    # Concatenate the u and v components for the right h
    u_name = f"u{h}"
    v_name = f"v{h}"

    # Perform weighted interpolation
    u_interp = (ds[u_name] * weights).sum(dim=('latitude', 'longitude'))
    v_interp = (ds[v_name] * weights).sum(dim=('latitude', 'longitude'))

    # Calculate wind direction
    wd = get_wd(u=u_interp, v=v_interp)

    # Calculate wind speed
    ws = get_ws(u=u_interp, v=v_interp)

    return ws, wd


def get_alpha(ws_z, ws_zr, z, zr):
    """
    Calculate shear exponent alpha in the power law relationship u(z)/u(zr) = (z/zr)^alpha.
    """
    # Calculate the ratio of u(z) to u(zr)
    ws_ratio = ws_z / ws_zr

    # Calculate the ratio of z to zr
    z_ratio = z / zr

    # Calculate alpha using the natural logarithm
    alpha = np.log(ws_ratio) / np.log(z_ratio)

    return alpha


def shear_extrapolation(hh, ws_zr, zr, alpha):
    """
    Calculate u(z) in the power law relationship u(z) = u(zr) * (z/zr)^alpha.
    """
    # Calculate the ratio of z to zr
    z_ratio = hh / zr

    # Calculate u(z) using the power law
    ws_hh = ws_zr * (z_ratio ** alpha)

    return ws_hh


def get_veer(wd_z1, wd_z2, z1, z2):
    # Calculate the difference in wind direction
    veer = wd_z2 - wd_z1

    # Ensure the veer is within the range [-180, 180]
    veer = ((veer + 180) % 360 - 180)/(z2 - z1)

    return veer


def extrapolate_wd(hh, wd_z1, z1, veer):
    """
    Extrapolate the wind direction to hub height given the wind veer.
    """
    # Calculate the change in height
    delta_z = hh - z1

    # Extrapolate the wind direction
    wd_hh = wd_z1 + veer * delta_z

    # Ensure the wind direction is within the range [0, 360)
    wd_hh = wd_hh % 360

    return wd_hh


def get_ws_wd_for_xyz(ds, hh, lat, lon, exponent):
    """
    Get ws and wd for lat, lon and hh within ds coords.
    """
    # horizontal interpolation for target latitude and longitude
    ws10, wd10 = get_uv_interpolated_ws_wd(ds, lat, lon, exponent, 10)
    ws100, wd100 = get_uv_interpolated_ws_wd(ds, lat, lon, exponent, 100)

    # vertical extrapolation for target hub height
    # ws
    alpha = get_alpha(ws10, ws100, 10, 100)
    ws_extra = shear_extrapolation(hh, ws100, 100, alpha)

    # wd
    veer = get_veer(wd10, wd100, 10, 100)
    wd_extra = extrapolate_wd(hh, wd10, 10, veer)

    return ws_extra, wd_extra


def wbl_fit(ws, plot=False):
    """
    Fit Weibull distribution for wind speed and optionally plot the histogram and Weibull fit.

    Parameters:
    ws (array-like): Array of wind speed data.
    plot (bool): Whether to plot the histogram and Weibull fit (default is False).

    Returns:
    tuple: A tuple containing the shape parameter (k) and scale parameter (A) of the fitted Weibull distribution.
    """
    def wbl_pdf(x, k, A):
        """
        Weibull probability density function.
        """
        return (k / A) * (x / A) ** (k - 1) * np.exp(-(x / A) ** k)

    # Compute the values if ws is a Dask array
    if hasattr(ws, 'compute'):
        ws = ws.compute()    
    # Convert ws to a numpy array if it is an xarray DataArray
    if hasattr(ws, 'values'):
        ws = ws.values

    # Filter out zero wind speeds to avoid division by zero
    ws = ws[ws > 0]

    # Create histogram data
    hist, bin_edges = np.histogram(ws, bins='auto', density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial guess for the parameters
    # k = 2 works for most N. Europe, for A try then gamma(1 + 1/2)≈0.88
    initial_guess = [2.0, np.mean(ws)/0.886]

    # Fit the Weibull distribution to the histogram data
    params, _ = curve_fit(wbl_pdf, bin_centers, hist, p0=initial_guess)

    k, A = params

    # Optionally plot the histogram and Weibull fit
    if plot:
        fig = plt.figure(figsize=(10, 6))
        plt.hist(ws, bins='auto', density=True, alpha=0.6, color='g', label='Histogram')

        # Generate Weibull PDF values for plotting
        x = np.linspace(0, np.max(ws), 100)
        y = wbl_pdf(x, k, A)

        plt.plot(x, y, 'r-', linewidth=2, label='Weibull Fit')
        plt.xlabel('Wind Speed')
        plt.ylabel('Probability Density')
        plt.title('Wind Speed Distribution with Weibull Fit')
        plt.legend()
        plt.grid(True)
        plt.show()

    if plot:
        return k, A, fig
    else:
        return k, A


def plot_wind_rose(wd, ws, h):
    """
    Plot a wind rose using the windrose package.
    """
    # Compute the values if wd or ws is a Dask array
    if hasattr(wd, 'compute'):
        wd = wd.compute()
    if hasattr(ws, 'compute'):
        ws = ws.compute()

    # Convert wd and ws to numpy arrays if they are xarray DataArrays
    if hasattr(wd, 'values'):
        wd = wd.values
    if hasattr(ws, 'values'):
        ws = ws.values

    # Create a new figure and add the windrose axes
    fig = plt.figure(figsize=(8, 8))
    ax = WindroseAxes.from_ax()

    # Plot the wind rose
    ax.bar(wd, ws, normed=True, opening=0.8, edgecolor="white")

    # Add a legend
    ax.set_legend()
    ax.set_title(f"Wind Frequency Rose at {h} m")

    return ax


def read_pc(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1,
                    encoding='windows-1252', usecols=(0, 1))
    wsp = data[:, 0]
    pow = data[:, 1]

    return wsp, pow


def get_power(wsp, pow, wsp_ts):
    '''
    Get power as f(wsp) for chosen power curve
    '''
    # Create interpolation function
    interp_func = interp1d(wsp, pow, kind='linear',
                        bounds_error=False, fill_value=0.0)
    # Interpolate power for the given wind speed time series
    pow_ts = interp_func(wsp_ts)

    return pow_ts


def get_gross_aep(file_path, wsp_ts):
    # Read power curve
    wsp, pow = read_pc(file_path)
    
    # Estimate power (and energy) timeseries
    pow_ts = get_power(wsp, pow, wsp_ts)
    
    # Get gross AEP
    tot_e = pow_ts.sum() * 1 # kWh # TODO: adapt for other time periods
    number_hrs = pow_ts.size * 1 # hrs
    avg_p = tot_e / number_hrs
    gross_aep = 365.25 * 24 * avg_p / 1e6 # GWh
    
    return gross_aep


#%% Case Analysis Class

class analysis_case:
    def __init__(self, power_curve_path, input_nc_folder, start_date, end_date):
        self.power_curve_path = power_curve_path
        self.input_nc_folder = input_nc_folder
        self.start_date = start_date
        self.end_date = end_date
        self.hubheight = None
        self.A = None
        self.k = None
        self.average_ws = None
        self.gross_aep = None

    def calculate(self, lat, lon, exponent, hubheight):
        self.hubheight = hubheight # attributes added only when calling method
        self.lat = lat
        self.lon = lon
        # Load NetCDF files
        ds = load_ncs(input_nc_folder=self.input_nc_folder,
                    start_date=self.start_date,
                    end_date=self.end_date)

        # Get wind speed and direction
        ws, wd = get_ws_wd_for_xyz(ds,
                                hh=self.hubheight,
                                lat=lat, lon=lon,
                                exponent=exponent)

        # Calculate Weibull parameters
        self.k, self.A = wbl_fit(ws)

        # Calculate average wind speed
        self.average_ws = float(np.mean(ws))

        # Calculate gross AEP
        self.gross_aep = get_gross_aep(self.power_curve_path, ws)

    def __str__(self):
        return (f"Analysis Case:\n"
                f"  Power Curve: {self.power_curve_path}\n"
                f"  Input NC Folder: {self.input_nc_folder}\n"
                f"  Start Date: {self.start_date}\n"
                f"  End Date: {self.end_date}\n"
                f"  Hub Height: {self.hubheight} m\n"
                f"  Latitude: {self.lat}°\n"
                f"  Longitude: {self.lon}°\n"
                f"  Wbl. A: {self.A:.2f} m/s\n"
                f"  Wbl. k: {self.k:.2f} -\n"
                f"  Avg. Wind Speed: {self.average_ws:.2f} m/s\n"
                f"  Gross AEP: {self.gross_aep:.1f} GWh")