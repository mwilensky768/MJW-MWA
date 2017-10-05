import pyuvdata

UV = pyuvdata.UVData()

UV.read_uvfits('/Users/mike_e_dubs/python_stuff/smaller_uvfits/f1061313008.uvfits')

UV.select(polarizations=[-5, ])

UV.write_uvfits('/Users/mike_e_dubs/python_stuff/smaller_uvfits/pf1061313008.uvfits')
