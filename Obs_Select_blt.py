import pyuvdata as pyuv
import time

UV1 = pyuv.UVData()
UV2 = pyuv.UVData()

print('I started reading! ' + time.strftime('%H:%M:%S'))

UV1.read_uvfits('/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits')
UV2.read_uvfits('/Users/mike_e_dubs/python_stuff/uvfits/1061313128.uvfits')

print('I finished reading! ' + time.strftime('%H:%M:%S'))

ind = []

for m in range(56):
    r = 8128 * m
    for n in range(100):
        ind.append(r + n)

UV1.select(blt_inds=ind)
UV2.select(blt_inds=ind)

print('I finished selecting! ' + time.strftime('%H:%M:%S'))

UV1.write_uvfits('/Users/mike_e_dubs/python_stuff/smaller_uvfits/s1061313008.uvfits')
UV2.write_uvfits('/Users/mike_e_dubs/python_stuff/smaller_uvfits/s1061313128.uvfits')

print('I finished writing! ' + time.strftime('%H:%M:%S'))
