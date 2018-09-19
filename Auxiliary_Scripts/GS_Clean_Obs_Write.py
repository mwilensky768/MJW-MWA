from SSINS import util
import glob

GS_file = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Obs_Lists/Golden_Set_OBSIDS.txt'
obslist = util.make_obslist(GS_file)

figpath = '/Users/mike_e_dubs/MWA/INS/Golden_Set/Golden_Set_COTTER_Filter_O3_S5/figs'
L = len(figpath)
dirty_obslist = glob.glob('%s/*data_match.png' % (figpath))
dirty_obslist = [obs[L + 1: L + 11] for obs in dirty_obslist]
for obs in dirty_obslist:
    obslist.remove(obs)
outfile = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Obs_Lists/GS_5sig_cubic_clean.txt'
util.make_obsfile(obslist, outfile)
