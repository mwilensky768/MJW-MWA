import imageio
import glob

inpath = '/Users/mike_e_dubs/MWA/FHD/Meteor_Trail/output_images/'
outpath = '/Users/mike_e_dubs/MWA/FHD/Meteor_Trail/'
duration = 0.2
im_list = glob.glob('%s*Residual_XX.png' % (inpath))
im_list.sort()
images = [imageio.imread(im) for im in im_list]
imageio.mimsave('%s1061313128_t11_t27.gif' % (outpath), images, duration=duration)
