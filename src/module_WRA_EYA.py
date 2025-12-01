#%% Module's imports
import xarray as xr
from glob import glob
import numpy as np
#import pandas as pd

#%% Module's Custom Functions
def load_ncs(input_nc_folder = '../inputs',
            start_date='1997-01-01',
            end_date='2008-12-31',
            chunks=None):
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


def derive_vel(ds):
    """
    Compute wind speed and wind direction based on u and v-components
    of wind at 10 and 100m. Add to input ds.
    """
    hs = [10, 100]

    for h in hs:
        # Concatenate the u and v names
        u_name = f"u{h}"
        v_name = f"v{h}"

        # Calculate wind speed
        ws = np.sqrt(ds[u_name]2 + ds[v_name]2)
        ws_name = f'ws{h}'
        # Add to ds
        ds[ws_name] = ws

        # Calculate wind direction
        wd_name = f'wd{h}'
        wd = (270 - np.degrees(np.arctan2(ds[v_name], ds[u_name]))) % 360
        # Add to ds
        ds[wd_name] = wd

    return ds


def ws_hor_interpolation(ds, lat_target, lon_target, exp=2):
    """
    Compute wind speed time series at 10 m and 100 m heights for a given
    location inside the box bounded by the 4 locations using interpolation.
    Option for inv. distance or dist. squared via exponent.
    """
    # Extract ws10 and ws100 from the dataset
    ws10 = ds['ws10']
    ws100 = ds['ws100']

    # Define auxiliary function
    def inv_dist_weighted_interpolation(ds_var, lat_target, lon_target, exponent):
        """
        Perform inverse distance weighted interpolation.
        """
        # Calculate the distances
        lat_delta = (ds_var['latitude'] - lat_target)
        lon_delta = (ds_var['longitude'] - lon_target)
        distances = (lat_delta  2 + lon_delta  2) ** 0.5

        # Avoid division by zero
        distances = distances.where(distances > 0, 1e-10)

        # Calculate weighting
        weights = 1 / (distances ** exponent)

        # Normalize weighting
        weights = weights / weights.sum()

        # Perform weighted interpolation
        interpolated_value = (ds_var * weights).sum(dim=('latitude', 'longitude'))

        return interpolated_value

    ws10_target = inv_dist_weighted_interpolation(ws10, lat_target, lon_target, exp)
    ws100_target = inv_dist_weighted_interpolation(ws100, lat_target, lon_target, exp)

    return ws10_target, ws100_target


def ws_vert_interpolation(z1, z2, zh, t_start=1997, t_end=1998):
    """Compute wind speed time series at height z for a given location
    inside the box bounded by the four locations using power law profile."""
    pass


def wbl_fit(wsp, t_start=1997, t_end=1998):
    """Fit Weibull distribution for wind speed at # a given location
    (inside the box) and a given height."""
    pass


def plot_wpsd(wsp, t_start=1997, t_end=1998):
    """Plot wind speed distribution (histogram vs. fitted Weibull
    distribution) at a given location (inside the box) and given height."""
    pass


def plot_wdir(wdir, t_start=1997, t_end=1998):
    """Plot wind rose diagram that shows the frequencies of different
    wind direction at a given location (inside the box) and given height."""
    pass


def gross_aep(wsp, pc):
    """Compute AEP of a specifed wind turbine (NREL 5 MW or NREL 15 MW)
    at a given location inside the box for a given year in the period we
    have provided the wind data."""
    pass

# Note that for the tasks listed in points 3-7, the functions should be able to
# specify the starting year (default to be 1997) and ending year (default to be
# 2008), thus defining which years' data to be used.

# %%