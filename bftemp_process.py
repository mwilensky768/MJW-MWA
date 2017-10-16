import numpy as np
import matplotlib.pyplot as plt

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

obslist = np.array(obslist).astype(int)  # Last line is blank so have to remove that from list before conversion
obslist = np.sort(obslist)  # Ensure the obslist is actually in sequence
lines = np.array(lines)
temp_avg = np.zeros(len(obslist), dtype=float)
obs_count = np.zeros(len(obslist), dtype=int)

lines = lines[[line.find('NULL') == -1 for line in lines]]
lines = lines[1:-2]  # remove header and footer

lines_obs = np.array([line[line.find('|') + 1:line.find('|') + 11] for line in lines]).astype(int)
good_lines_obs = lines_obs[np.logical_and(min(obslist) < lines_obs, lines_obs < max(obslist))]
good_lines = lines[np.logical_and(min(obslist) < lines_obs, lines_obs < max(obslist))]
good_lines_temps = np.array([line[line.find('{') + 1:line.find('}')].split(",")
                             for line in good_lines]).astype(float)

avg_1 = 0.125 * np.sum(good_lines_temps, axis=1)

for k in range(len(obslist) - 1):
    obs_count[k] = len(good_lines_obs[np.logical_and(obslist[k] < good_lines_obs,
                                                     good_lines_obs < obslist[k + 1])])

m = 0

for k in range(len(obs_count)):
    if obs_count[k] > 0:
        steps = min(112, obs_count[k])
        temp_avg[k] = np.sum(avg_1[m:m + steps]) / steps  # 16 receivers, 16 seconds b/w measurements, 112s per obs
    m += obs_count[k]

np.save('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_Avg_Obs_Temp.npy', temp_avg)
np.save('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_Temp_Obs_Count.npy', obs_count)

print('The average bftemp is ' + str(np.mean(temp_avg[temp_avg > 0])))
print('The number of long run observations in this file is ' + str(len(obs_count[obs_count > 0])))
