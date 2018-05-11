import pyuvdata

UV = pyuvdata.UVData()
UV.read_uvfits('/Users/mike_e_dubs/MWA/Data/uvfits/1061313128.uvfits', read_data=False)
print(UV.Nbls)
print(UV.Ntimes)
