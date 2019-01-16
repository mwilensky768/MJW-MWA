import pickle
import numpy as np
from SSINS import util as u

with open('/Users/mike_e_dubs/MWA/INS/Long_Run/Original_Jackknife_Revamp_Complete/TV_bright_dict_total.pik') as file:
    TV_bright_dict = pickle.load(file)

bright_list = [TV_bright_dict[obs] for obs in TV_bright_dict]
sorted_bright_list = sorted(bright_list)

top_50 = [obs for obs in TV_bright_dict if (TV_bright_dict[obs] > sorted_bright_list[-51])]
top_50_bright_list = [TV_bright_dict[obs] for obs in TV_bright_dict if (TV_bright_dict[obs] > sorted_bright_list[-51])]
print(sorted(top_50_bright_list) == sorted_bright_list[-50:])

top_100 = [obs for obs in TV_bright_dict if (TV_bright_dict[obs] > sorted_bright_list[-101])]
top_100_bright_list = [TV_bright_dict[obs] for obs in TV_bright_dict if (TV_bright_dict[obs] > sorted_bright_list[-101])]
print(sorted(top_100_bright_list) == sorted_bright_list[-100:])

top_200 = [obs for obs in TV_bright_dict if (TV_bright_dict[obs] > sorted_bright_list[-201])]
top_200_bright_list = [TV_bright_dict[obs] for obs in TV_bright_dict if (TV_bright_dict[obs] > sorted_bright_list[-201])]
print(sorted(top_200_bright_list) == sorted_bright_list[-200:])

u.make_obsfile(top_50, '/Users/mike_e_dubs/MWA/INS/Long_Run/Original_Jackknife_Revamp_Complete/TV_obs_top50.txt')
u.make_obsfile(top_100, '/Users/mike_e_dubs/MWA/INS/Long_Run/Original_Jackknife_Revamp_Complete/TV_obs_top100.txt')
u.make_obsfile(top_200, '/Users/mike_e_dubs/MWA/INS/Long_Run/Original_Jackknife_Revamp_Complete/TV_obs_top200.txt')
