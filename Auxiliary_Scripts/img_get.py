import requests
import time
import os

obslist_dir = '/Users/mike_e_dubs/MWA/Obs_Lists/'
GC_dir = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/'

GS_outdir = GC_dir + 'Golden_Set_8s_Autos/Skymaps/'
LR_outdir = GC_dir + 'Long_Run_8s_Autos/Skymaps/'
S2_outdir = GC_dir + 'S2_Zenith_8s_Autos/Skymaps/'
DS_outdir = GC_dir + 'Diffuse_2015_8s_Autos/Skymaps/'

GS_List = obslist_dir + 'Golden_Set_OBSIDS.txt'
LR_List = obslist_dir + 'Long_Run_8s_Autos_OBSIDS.txt'
S2_List = obslist_dir + 'season2_zenith_calcut.txt'
DS_List = obslist_dir + 'sidelobe_survey_obsIDs.txt'

txt_list = [GS_List, LR_List, S2_List, DS_List]
outdir_list = [GS_outdir, LR_outdir, S2_outdir, DS_outdir]

baseurl = 'http://mwa-metadata01.pawsey.org.au/observation/skymap/?obs_id='

for k in range(3, 4):
    with open(txt_list[k]) as f:
        obslist = f.read().split("\n")

    print(k)

    if k == 3:
        obslist = obslist[885:]

    for obs in obslist:
        img_data = requests.get(baseurl + obs).content
        with open('%s%s_skymap.png' % (outdir_list[k], obs), 'wb') as handler:
            handler.write(img_data)

        time.sleep(60)

    time.sleep(60)
