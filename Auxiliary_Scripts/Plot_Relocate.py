import glob
import shutil
import os

plot_dir = '/Users/mike_e_dubs/MWA/Catalogs/Wenyang_Phase2/data_eva/unflagged/'
target_dir = '/Users/mike_e_dubs/MWA/Catalogs/Wenyang_Phase2/data_eva/frac_diff/'

plots = glob.glob('%s*__INS_frac_diff.png' % (plot_dir))
print(plots)

for plot in plots:
    shutil.copy(plot, target_dir)
