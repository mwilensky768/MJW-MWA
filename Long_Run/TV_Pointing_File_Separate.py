import shutil
import os

pointings = ['Minus_2', 'Minus_1', 'Zenith', 'Plus_1', 'Plus_2']
list_base = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Obs_Lists/Long_Run'
plot_base = '/Users/mike_e_dubs/MWA/INS/Long_Run'
for pointing in pointings:
    with open('%s/%s_TV_4.5.txt' % (list_base, pointing)) as f:
        obslist = f.read().split("\n")
    obslist.remove('')
    outdir = '%s/TV_4.5/%s' % (plot_base, pointing)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for obs in obslist:
        shutil.copy('%s/%s_Match_Filter_Only/figs/%s_spw0_INS_ms_match.png' %
                    (plot_base, pointing, obs), outdir)
