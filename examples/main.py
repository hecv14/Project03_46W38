#%% imports
import os
import xarray as xr

#%% open nc
folder_inputs = "../inputs"
filename_nc = "1997-1999.nc"
filepath = os.path.join(folder_inputs, filename_nc)

ds = xr.open_dataset(filepath)
print(ds)

# %%
u10 = ds['u10'].values

# %%
