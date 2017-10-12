import numpy as np

obslist_path = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_8s_Autos_OBSIDS.txt'
cutlist_path = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_8s_Autos_Funky_OBSIDS.txt'
bftemps_path = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/bftemps.txt'

with open(obslist_path) as f:
    obslist = f.read().split("\n")
with open(cutlist_path) as g:
    cutlist = g.read().split("\n")
with open(bftemps_path) as h:
    lines = h.read().split("\n")

for item in cutlist:
    obslist.remove(item)

flags = np.zeros(len(lines), dtype=bool)

for k in range(len(lines)):
    if lines[k][2:12] in obslist:
        flags[k] += 1

good_lines = lines[flags > 0]
