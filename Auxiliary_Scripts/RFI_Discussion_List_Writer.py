import glob
import shutil

LR_fm_list = glob.glob('/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Print/Flags/GS/*.png')
LR_list = [path[path.find('GS/') + 3:path.find('GS/') + 13] for path in LR_fm_list]

LR_va_list = glob.glob('/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Unflagged/*.png')
#LR_ft_list = glob.glob('/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Diffuse_2015_8s_Autos/Freq_Time/*.png')
#LR_sm_list = glob.glob('/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Diffuse_2015_8s_Autos/Skymaps/*.png')


LR_va_list = [path for path in LR_va_list if
              path[path.find('Golden_Set_8s_Autos/') + 38:path.find('Golden_Set_8s_Autos/') + 48]
              in LR_list]
#LR_ft_list = [path for path in LR_ft_list if
              #path[path.find('Diffuse_2015_8s_Autos/') + 32:path.find('Diffuse_2015_8s_Autos/') + 42] in LR_list]
#LR_sm_list = [path for path in LR_sm_list if
              #path[path.find('Diffuse_2015_8s_Autos/') + 30:path.find('Diffuse_2015_8s_Autos/') + 40] in LR_list]

for fig in LR_va_list:
    shutil.copy(fig, '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Print/Vis_Avg/GS/Unflagged/')
#for fig in LR_ft_list:
    #shutil.copy(fig, '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Print/Freq_Time/DS/')
#for fig in LR_sm_list:
    #shutil.copy(fig, '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Print/Skymaps/DS/')
