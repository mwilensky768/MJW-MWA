from SSINS import util as u
import glob

LR = u.make_obslist('/Users/mike_e_dubs/Repositories/MJW-MWA/Obs_Lists/Long_Run_8s_Autos_OBSIDS.txt')

time_dir = '/Users/mike_e_dubs/MWA/INS/Long_Run/time_arrs'
t_list = glob.glob('%s/*lst_arr.npy' % time_dir)
t_list = [path[len(time_dir) + 1:len(time_dir) + 11] for path in t_list]
print(len(t_list))

arr_dir = '/Users/mike_e_dubs/MWA/INS/Long_Run/Original/metadata'
arr_list = glob.glob('%s/*pols.npy' % arr_dir)
arr_list = [path[len(arr_dir) + 1:len(arr_dir) + 11] for path in arr_list]
print(len(arr_list))

missing_obs = []
for obs in LR:
    if (obs not in t_list) or (obs not in arr_list):
        missing_obs.append(obs)

u.make_obsfile(missing_obs, '/Users/mike_e_dubs/MWA/INS/Long_Run/Original/missing_obs_AWS.txt')
