import xarray as xr
ds = xr.open_dataset("data/era5_cyclone_dikeledi_700hPa.nc")
print(ds)