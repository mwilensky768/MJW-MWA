from SSINS import SS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('inpath')
args = parser.parse_args()

read_kwargs = {'file_type': 'ms'}

ss = SS(inpath=args.inpath, read_kwargs=read_kwargs)
