from pyuvdata import UVData
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("inpath", action='store', nargs=1, help="The file you want to process")
args = parser.parse_args()

UV = UVData()
UV.read_uvfits(args.inpath[0])

ind = np.where(UV.nsample_array == 0)
for m in range(len(ind[0])):
    UV.nsample_array[ind[0][m], ind[1][m], ind[2][m], ind[3][m]] = \
        UV.nsample_array[0, ind[1][m], ind[2][m] % 16, ind[3][m]]

UV.flag_array([:, :, :163, :]) = 1
UV.flag_array([:, :, 163, :]) = 0
UV.flag_array([:, :, 164:, :]) = 1

UV.write_uvfits(args.inpath[0])
