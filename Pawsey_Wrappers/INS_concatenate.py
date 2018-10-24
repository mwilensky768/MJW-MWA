from SSINS.INS_helpers import INS_concat
from SSINS import util
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('obs')
parser.add_argument('outdir')
args = parser.parse_args()

INS_sequence = []
axis = 2
for i in range(24):
    read_paths = util.read_paths('%s_box%i' % (outdir, i), None, args.obs, 'INS')
    INS_sequence.append(INS(read_paths=read_paths, obs=args.obs, outpath=args.outdir))
metadata_kwargs = {'vis_units': INS_sequence[0].vis_units,
                   'pols': INS_sequence[0].pols}
ins = INS_concat(INS_sequence, axis=2 metadata_kwargs=metadata_kwargs)
ins.save()
