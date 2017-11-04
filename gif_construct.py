from images2gif import writeGif
from PIL import Image
import glob

inpath = '/Users/mike_e_dubs/MWA/Catalogs/Golden_Set_8s_Autos/Ant_Scatter/Narrowband/'
outpath = '/Users/mike_e_dubs/MWA/Animations/Golden_Set_8s_Autos/Ant_Scatter/Narrowband/'

obslist_path = '/Users/mike_e_dubs/MWA/Obs_Lists/Golden_Set_Narrowband_OBSIDS.txt'
with open(obslist_path) as f:
    obslist = f.read().split("\n")

for obs in obslist:
    im_list = glob.glob(inpath + obs + '*')
    images = [Image.open(im) for im in im_list]
    writeGif(outpath + obs + '.gif', images, duration=30)
