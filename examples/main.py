#%% Imports
import sys
sys.path.append('../src')

from module_WRA_EYA import load_ncs, derive_vel, ws_hor_interpolation

# %%
# load_ncs
ds = load_ncs()

# add ws and wd
derive_vel(ds)

# %%
lat = 55.6
lon = 7.9
ws10_target, ws100_target = ws_hor_interpolation(ds, lat, lon)

# %%