import numpy as np
from SSINS import plot_lib as pl
import matplotlib.pyplot as plt
import os

Soft_List = [1061314960, 1061658392, 1061659000, 1061659368, 1061659976,
             1062175376, 1062348560, 1062521128, 1062779504, 1062779624,
             1063123792, 1063641632, 1064589072, 1064589432, 1064589560,
             1064933120, 1064934216, 1065277528, 1065449432, 1066569560,
             1066571152, 1066743600, 1067088008, 1067259000, 1067259848,
             1067260336, 1067260464, 1062949760, 1063466744, 1064931896,
             1065276184, 1066568584, 1066741280, 1066741400, 1067084960,
             1067085936, 1067086056, 1067086304, 1061318008, 1061662296,
             1062351728, 1062523816, 1062525160, 1063126840, 1063644928,
             1063645168, 1063645288, 1064764328, 1064764568, 1064936536,
             1064937264, 1067091552, 1068814344, 1068814584, 1068814832,
             1069762784, 1063128792, 1063130136, 1064939096, 1067092280,
             1061315808, 1061316296, 1062351000, 1062522472, 1062522592,
             1062954024, 1063126480, 1064763104, 1064763472, 1064763592,
             1064763720, 1064935680, 1065279112, 1066743968, 1067089472]

Hard_List = [1064589432, 1064761640, 1064933240, 1067087888, 1064932016,
             1065275336, 1062352216, 1064764448, 1064765304, 1066573952,
             1067090696, 1063128672, 1064765912, 1064766648, 1068816176,
             1062350880, 1064934944, 1066572976]

Streak_List = [1062176352, 1062176840, 1062348560, 1062779504, 1063123792,
               1065449432, 1066569560, 1066741888, 1066743600, 1067259000,
               1067259120, 1067259848, 1067260096, 1067260336, 1068810680,
               1068810928, 1065878416, 1067257048, 1067257776, 1067257896,
               1067258752, 1062351728, 1066573952, 1063128552, 1063128792,
               1062780968, 1062781088, 1063126600, 1067089472, 1068811904]
titles = ['Soft', 'Hard', 'Streak']
labels = ['asc', 'desc', 'clump']
outdir = '/Users/mike_e_dubs/MWA/Test_Plots/Clump_Sum'
if not os.path.exists(outdir):
    os.makedirs(outdir)

for k, lst in enumerate([Soft_List, Hard_List, Streak_List]):
    asc_sum_total = []
    desc_sum_total = []
    clump_sum_total = []
    for obs in lst:
        if os.path.exists('/Users/mike_e_dubs/MWA/INS/Long_Run/All/Match_Filter/arrs/%s_None_INS_match_events.npy' % obs):
            events = np.load('/Users/mike_e_dubs/MWA/INS/Long_Run/All/Match_Filter/arrs/%s_None_INS_match_events.npy' % obs)
            asc_sum = 0 # low value means it was caught in ascending order
            desc_sum = 0 # low value means it was caught in descending order
            clump_sum = np.count_nonzero(np.absolute(np.diff(events[:, -1])) > 1) # low value here means it was caught in clumps
            for i, t in enumerate(events[:, -1]):
                asc_sum += np.count_nonzero(events[i:, -1] < t)
                desc_sum += np.count_nonzero(events[i:, -1] > t)
            asc_sum_total.append(asc_sum)
            desc_sum_total.append(desc_sum)
            clump_sum_total.append(clump_sum)
    asc_hist = np.histogram(asc_sum_total, bins='auto')
    desc_hist = np.histogram(desc_sum_total, bins='auto')
    clump_hist = np.histogram(clump_sum_total, bins='auto')
    if k == 1:
        print(clump_hist[0])
    assert(np.sum(asc_hist[0]) == np.sum(desc_hist[0]))
    assert(np.sum(asc_hist[0]) == np.sum(clump_hist[0]))
    fig, ax = plt.subplots(figsize=(14, 8))
    for m, hist in enumerate([asc_hist, desc_hist, clump_hist]):
        x = hist[1][:-1] + 0.5 * np.diff(hist[1])
        pl.error_plot(fig, ax, x, hist[0], xlabel='Sum Total', ylabel='counts',
                      label=labels[m], title='Lone Run Order Parameter Histogram, %s' % titles[k])
    fig.savefig('%s/%s.png' % (outdir, titles[k]))
