import imageio
import glob

inpath = '/Users/mike_e_dubs/MWA/Catalogs/S2_Zenith_Calcut_8s_Autos/Ant_Scatter/Chirp/'
outpath = '/Users/mike_e_dubs/MWA/Animations/S2_Zenith_Calcut_8s_Autos/Ant_Scatter/Chirp/'

obslist_path = '/Users/mike_e_dubs/MWA/Obs_Lists/S2_Zenith_Calcut_8s_Autos_Chirp.txt'
with open(obslist_path) as f:
    obslist = f.read().split("\n")

pols = ['XX', 'YY', 'XY', 'YX']
im_list = {}
for obs in obslist:
    for k in range(4):
        im_list = sorted(glob.glob('%s%s*pol%i*' % (inpath, obs, k)),
                         key=lambda name: int(name[name.find('_t') + 2:-4]))
        images = [imageio.imread(im) for im in im_list]
        if len(images) > 0:
            imageio.mimsave('%s%s_%s.gif' % (outpath, obs, pols[k]), images, duration=0.5)
