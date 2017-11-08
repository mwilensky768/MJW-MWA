import pyuvdata

UV = pyuvdata.UVData()

UV.read_uvfits('/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits')

UV.select(freq_chans=[300, 301, 302])

UV.write_uvfits('/Users/mike_e_dubs/python_stuff/smaller_uvfits/f1061313008.uvfits')
