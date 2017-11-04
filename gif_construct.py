import imageio
import glob

inpath = '/Users/mike_e_dubs/MWA/Catalogs/Golden_Set_8s_Autos/Ant_Scatter/Narrowband/'
outpath = '/Users/mike_e_dubs/MWA/Animations/Golden_Set_8s_Autos/Ant_Scatter/Narrowband/'

obslist_path = '/Users/mike_e_dubs/MWA/Obs_Lists/Golden_Set_Narrowband_OBSIDS.txt'
with open(obslist_path) as f:
    obslist = f.read().split("\n")

for obs in obslist:
    im_list = sorted(glob.glob(inpath + obs + '*'), key=lambda name: int(name[104:-4]))
    images = [imageio.imread(im) for im in im_list]
    imageio.mimsave(outpath + obs + '.gif', images, duration=30)
