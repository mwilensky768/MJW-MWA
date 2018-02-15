import shutil
import glob

fig_dir = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Diffuse_2015_8s_Autos/Vis_Avg/Unflagged/'

fig_path_list = glob.glob('%s*.png' % (fig_dir))
fig_path_list_rename = ['%s_Post_Flag.png' % (path[:-4]) for path in fig_path_list]

for path, path_r in zip(fig_path_list, fig_path_list_rename):
    shutil.copy(path, path_r)
