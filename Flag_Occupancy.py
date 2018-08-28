from SSINS import SS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('obs')
parser.add_argument('inpath')
parser.add_argument('outpath')
parser.parse_args()

sky_sub = SS(obs=args.obs, outpath=args.outpath, inpath=args.inpath,
             bad_time_indices=(0, -1, -2, -3), read_kwargs={'ant_str': 'cross'})

sky_sub.INS_prepare()
sky_sub.save_data()
sky_sub.apply_flags(choice='original')
sky_sub.INS_prepare()
sky_sub.save_data()
