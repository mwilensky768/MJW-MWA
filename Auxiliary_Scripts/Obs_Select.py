import pyuvdata as pyuv
import time

UV1 = pyuv.UVData()
UV2 = pyuv.UVData()

print('I started reading! ' + time.strftime('%H:%M:%S'))

UV1.read_uvfits('/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits')
UV2.read_uvfits('/Users/mike_e_dubs/python_stuff/uvfits/1061313128.uvfits')


print('I finished reading! ' + time.strftime('%H:%M:%S'))

ant_pairs1 = [UV1.baseline_to_antnums(k) for k in UV1.baseline_array[0:100]]
ant_pairs2 = [UV2.baseline_to_antnums(k) for k in UV2.baseline_array[0:100]]

UV1.select(ant_pairs_nums=ant_pairs1)
UV2.select(ant_pairs_nums=ant_pairs2)

print('I finished selecting! ' + time.strftime('%H:%M:%S'))

UV1.write_uvfits('/Users/mike_e_dubs/python_stuff/smaller_uvfits/ss1061313008.uvfits')
UV2.write_uvfits('/Users/mike_e_dubs/python_stuff/smaller_uvfits/ss1061313128.uvfits')

print('I finished writing! ' + time.strftime('%H:%M:%S'))
