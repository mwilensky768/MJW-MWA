from pyuvdata import UVData
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('dir1')
parser.add_argument('dir2')
args = parser.parse_args()

filelist1 = glob.glob('%s/vis_data/*' % args.dir1) + glob.glob('%s/metadata/*' % args.dir1)
filelist2 = glob.glob('%s/vis_data/*' % args.dir2) + glob.glob('%s/metadata/*' % args.dir2)

uv1 = UVData()
uv2 = UVData()

print(filelist1)
print(filelist2)

uv1.read_fhd(filelist=filelist1, use_model=True)
uv2.read_fhd(filelist=filelist2, use_model=True)

assert not np.all(uv1.data_array == uv2.data_array), "All the data are equal"

print("The obsdata are unique")
