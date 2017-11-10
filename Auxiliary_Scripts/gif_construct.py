import imageio
import glob

inpath = '/Users/mike_e_dubs/MWA/Catalogs/Golden_Set_8s_Autos/Vis_Avg/Narrowband/'
outpath = '/Users/mike_e_dubs/MWA/Animations/Golden_Set_8s_Autos/Vis_Avg/Narrowband/'
obslist_path = '/Users/mike_e_dubs/MWA/Obs_Lists/Golden_Set_Narrowband_OBSIDS.txt'
duration = 0.2

with open(obslist_path) as f:
    obslist = f.read().split("\n")
obslist.remove('')

im_list = {}
for obs in obslist:
    im_list = sorted(glob.glob('%s%s*' % (inpath, obs)),
                     key=lambda name: int(name[name.find('_t') + 2:-4]))
    images = [imageio.imread(im) for im in im_list]
    if len(images) > 0:
        imageio.mimsave('%s%s.gif' % (outpath, obs), images,
                        duration=duration)
