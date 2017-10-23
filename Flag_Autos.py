import pyuvdata

inpath = '/Users/mike_e_dubs/MWA/Data/smaller_uvfits/f1061313008.uvfits'  # uvfits file to read in
outpath = '/Users/mike_e_dubs/MWA/Data/smaller_uvfits/Af10611313008.uvfits'  # uvfits file to writeout

# Read in the stuff
UV = pyuvdata.UVData()
UV.read_uvfits(inpath)

# Find which baseline-time indices correspond to autos
blt_auto_inds = [k for k in range(UV.Nblts) if UV.ant_1_array[k] == UV.ant_2_array[k]]

# Set the flags to True
for ind in blt_auto_inds:
    UV.flag_array[ind, :, :, :] = 1

# Write out the new uvfits files with desired flags
UV.write_uvfits(outpath)
