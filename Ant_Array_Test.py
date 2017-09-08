import pyuvdata as pyuv

UV = pyuv.UVData()

UV.read_uvfits('/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits')

print(UV.ant_1_array[0:2 * 8128])
print(UV.ant_2_array[0:2 * 8128])
