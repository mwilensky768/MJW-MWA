import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('s')
args = parser.parse_args()

print(args.s)
np.save('%s/poop.npy' % args.s, [3, 4])
