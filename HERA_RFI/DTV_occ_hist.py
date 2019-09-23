import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
from SSINS.util import make_obslist

parser = argparse.ArgumentParser()
parser.add_argument('--yml_file_list', help='The yml files to operate on')
parser.add_argument('--outdir', help="The output directory")
args = parser.parse_args()

yml_list = make_obslist(args.yml_file_list)

inner_keys = ['occ', 'autopow', 'crosspow']
chan_keys = ['TV%i' % chan for chan in [4, 5, 6]]
list_dict = {chan_key: {key: [] for key in inner_keys} for chan_key in chan_keys}

for yml in yml_list:
    with open(yml, 'r') as yml_file:
        obsdict = yaml.load(yml_file, Loader=yaml.Loader)
    for chan in chan_keys:
        for key in inner_keys:
            list_dict[chan][key].append(obsdict[chan][key])

Titles = ['Occupancy per Obs', 'Average Flagged Autocorrelation Per Obs', 'Average Flagged Cross-Correlation Per Obs']
colors = ['organge', 'green', 'blue']

for key, title in zip(inner_keys, Titles):
    plt.hist([list_dict[chan][key] for chan in chan_keys], bins='auto', title=title,
             label=chan_keys, histtype='step', color=colors)
    for channel, color in zip(chan_keys, colors):
        plt.axvline(x=np.mean(list_dict[channel][key]), color='color')
    plt.savefig('%s/HERA_DTV_%s_hist.pdf' % (args.outdir, key))
