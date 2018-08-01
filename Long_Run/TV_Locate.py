import numpy as np
import glob

base = '/Users/mike_e_dubs/MWA/INS/Long_Run'
pointings = ['Minus_2', 'Minus_1', 'Zenith', 'Plus_1', 'Plus_2']
for pointing in pointings:
    f = open('/Users/mike_e_dubs/python_stuff/MJW-MWA/Obs_Lists/Long_Run/%s_TV_4.5.txt' % pointing, 'w')
    L = len('%s/%s_Match_Filter_Only/arrs/' % (base, pointing))
    event_arr_paths = glob.glob('%s/%s_Match_Filter_Only/arrs/*match_events.npy' % (base, pointing))
    for path in event_arr_paths:
        event_arr = np.load(path)
        event_arr_remodel = np.array([event for event in event_arr if
                                      event[1] != slice(None) and
                                     (event[1].indices(384)[1] - event[1].indices(384)[0]) > 1])
        if len(event_arr_remodel):
            f.write('%s\n' % path[L:L + 10])
