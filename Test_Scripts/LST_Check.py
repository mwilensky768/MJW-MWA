import pyuvdata
import glob
import numpy as np

pathlist = glob.glob('/data6/HERA/data/2458042/*.uv')

UV0 = pyuvdata.UVData()
UV48 = pyuvdata.UVData()
UV55 = pyuvdata.UVData()

UV0.read_miriad(pathlist[0])
UV48.read_miriad(pathlist[48])
UV55.read_miried(pathlist[55])

begin = UV0.lst_array[0]

ev_beg = UV48.lst_array[0]
end = UV55.lst_array[-1]

lst = [begin, ev_beg, end]
lst_len = np.diff(lst)

print('The LSTs of the events were: ' + str(lst))
print('The differences between the events were ' + str(list_len))
