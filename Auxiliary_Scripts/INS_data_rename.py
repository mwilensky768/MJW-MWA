import shutil
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('inpath', action='store', help='The path with the files to be moved')
args = parser.parse_args()

arrs = glob.glob('%s/arrs/INS/*' % args.inpath)
for arr in arrs:
    L = len('%s/arrs/INS/' % args.inpath)
    obs = arr[L:L + 10]
    if 'INS' in arr[L:]:
        shutil.move(arr, '%s/arrs/%s_None_INS_data.npym' % (args.inpath, obs))
    if 'Nbls' in arr[L:]:
        shutil.move(arr, '%s/arrs/%s_None_INS_Nbls.npym' % (args.inpath, obs))
