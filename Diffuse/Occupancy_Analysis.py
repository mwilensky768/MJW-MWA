import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
import csv


def print_N_zero(occ_dict, shape, obslist):
    subdict = occ_dict[5][shape]
    edges = [16 * i for i in range(24)] + [15 + 16 * i for i in range(24)]
    bool_ind = np.ones(384, dtype=bool)
    bool_ind[edges] = 0
    if shape != 'total':
        N0 = len([obs for obs in obslist if subdict[obs] == 0])
    else:
        print(np.amin([subdict[obs][bool_ind].mean() for obs in obslist]))
        N0 = len([obs for obs in obslist if subdict[obs][bool_ind].mean() == 0])
    print('The number of observations with zero occupancy is %i ' % N0)


def make_occ_hist(occ_dict, shape, obslist):
    subdict = occ_dict[5][shape]
    bins = np.linspace(0, 1, num=51)
    kwargs = {'bins': bins, 'histtype': 'step', 'density': True}

    if shape == 'total':
        edges = [16 * i for i in range(24)] + [15 + 16 * i for i in range(24)]
        occlist = np.ma.masked_array([subdict[obs] for obs in obslist])
        occlist[:, edges] = np.ma.masked
        fig, ax = plt.subplots(2, 1)
        ax[0].hist(occlist[np.logical_not(occlist.mask)].flatten(), label='Occupancy per Channel', **kwargs)
        ax[0].hist(np.mean(occlist, axis=1), label='Total Occupancy per Obs', **kwargs)
        ax[1].plot(np.mean(occlist, axis=0), label='Average Occupancy Per Channel')
        ax[0].set_xlabel('Occupancy')
        ax[0].set_ylabel('Density')
        ax[1].set_xlabel('Coarse Channel #')
        ax[1].set_ylabel('Occupancy')
        ax[0].legend()
        ax[1].legend()
    else:
        occlist = [subdict[obs] for obs in obslist]
        fig, ax = plt.subplots()
        ax.hist(occlist, label='%s Occupancy per Obs' % shape, **kwargs)
        ax.set_xlabel('Occupancy')
        ax.set_ylabel('Density')

    return(fig, ax)


def make_csvdata(occ_dict, shape, obslist):
    subdict = occ_dict[5][shape]
    edges = [16 * i for i in range(24)] + [15 + 16 * i for i in range(24)]
    bool_ind = np.ones(384, dtype=bool)
    bool_ind[edges] = 0
    if shape == 'total':
        csvdata = [[obs, subdict[obs][bool_ind].mean()] for obs in obslist]
    else:
        csvdata = [[obs, subdict[obs]] for obs in obslist]

    return(csvdata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dict', required=True, help='Outpath to dictionary')
    parser.add_argument('--obslist')
    parser.add_argument('-s', '--shape', required=True, help='Shape to process')
    parser.add_argument('-o', '--outdir', required=True)
    parser.add_argument('-t', '--tag', default='')
    parser.add_argument('-c', '--csv', action='store_true')
    args = parser.parse_args()

    with open(args.dict, 'rb') as file:
        occ_dict = pickle.load(file)
    if args.obslist is not None:
        obslist = args.obslist
    else:
        obslist = occ_dict[5][args.shape].keys()
    shape = args.shape

    func_args = (occ_dict, shape, obslist)

    print_N_zero(*func_args)
    fig, ax = make_occ_hist(*func_args)
    fig.savefig('%s/%s_occupancy.pdf' % (args.outdir, args.tag))
    if args.csv:
        csvdata = make_csvdata(*func_args)
        with open('%s/%s_occ.csv' % (args.outdir, args.tag), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csvdata)
    csvfile.close()
