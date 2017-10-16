import numpy as np
import matplotlib.pyplot as plt

obslist_path = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_8s_Autos_OBSIDS.txt'
cutlist_path = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_8s_Autos_Funky_OBSIDS.txt'

with open(obslist_path) as f:
    obslist = f.read().split("\n")
with open(cutlist_path) as g:
    cutlist = g.read().split("\n")

for item in cutlist:
    obslist.remove(item)

obslist_diff = np.diff(np.array(obslist).astype(int))

plt.scatter(obslist_diff, obslist_diff)

plt.show()
