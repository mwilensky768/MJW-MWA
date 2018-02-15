import shutil

obs_list_dir = '/Users/mike_e_dubs/MWA/Obs_Lists/'
sumit_cat_dir = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Sumit_Catalog/'
survs = ['DS_2015', 'GS_2013', 'LR_2013', 'S2_2014']
flag_slices = ['All', 'Post_Flag']


for surv in survs:
    with open('%s%s_Notable_Cases.txt' % (obs_list_dir, surv)) as f:
        obslist_notes = f.read().split("\n")

    while '' in obslist_notes:
        obslist_notes.remove('')

    obslist = [note[0:10] for note in obslist_notes]

    for obs in obslist:
        for flag_slice in flag_slices:
            shutil.copy('%sAll_Data/%s/%s_INS_%s.png' %
                        (sumit_cat_dir, surv, obs, flag_slice),
                        '%sNotable_Cases/%s/' % (sumit_cat_dir, surv))
