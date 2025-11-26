def read_netcdf():
    """Load and parse multiple provided netCDF4 files."""
    pass

def derive_vel(u, v):
    """Compute wind speed and wind direction based on u and v-components
    of wind."""
    pass

def horiz_extrap(x, y, t_start=1997, t_end=1998):
    """Compute wind speed and wind direction time series at 10 m and
    100 m heights for a given location inside the box bounded by the
    four locations, such as the Horns Rev 1 site, using interpolation."""
    pass

def vert_extrap(z1, z2, zh, t_start=1997, t_end=1998):
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
