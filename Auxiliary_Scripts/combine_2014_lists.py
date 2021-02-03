from SSINS.util import make_obslist, make_obsfile
import os

master_list = []
rootdir = '/Users/mike_e_dubs/MWA/Obs_Lists'

for pointing in ['minus_two', 'minus_one', 'zenith', 'plus_one', 'plus_two']:
    pointing_list = []
    for page_ind in [1, 2, 3]:
        path = '%s/2014_EoR_High_RA_350_pointing_%s_obs_page_%i.txt' % (rootdir, pointing, page_ind)
        if os.path.exists(path):
            pointing_page_list = make_obslist(path)
            pointing_list += pointing_page_list
    print(pointing_list)

    master_list += pointing_list

    make_obsfile(pointing_list, '%s/2014_EoR_High_RA_350_Pointing_%s_obs.txt' % (rootdir, pointing))

make_obsfile(master_list, '%s/2014_EoR_High_RA_350_Master_obs.txt' % rootdir)
