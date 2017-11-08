import pyuvdata
import numpy as np
import csv

UV = pyuvdata.UVData()

UV.read_uvfits('/Users/mike_e_dubs/python_stuff/smaller_uvfits/s1061313008.uvfits')

keys = ['Nfreqs', 'Npols', 'Ntimes', 'channel_width', 'integration_time']

properties = {}

for key in keys:
    properties[key] = getattr(UV, key)

properties['band_min'] = np.amin(UV.freq_array)
properties['band_max'] = np.amax(UV.freq_array)

Ncoarse = 24
Nfine = 16

LEdges = np.array([UV.freq_array[0, Nfine * k] for k in range(Ncoarse)])
REdges = np.array([UV.freq_array[0, (Nfine - 1) + Nfine * k] for k in range(Ncoarse)])
Centers = 0.5 * (REdges + LEdges)

properties['LEdges'] = LEdges
properties['REdges'] = REdges
properties['Centers'] = Centers
Fs = 655360000

Pre_Select = []
N_Pre = 256

for n in range(N_Pre):
    Pre_Select.append(float(n) / N_Pre * Fs / 2)

Pre_Select = np.array(Pre_Select)

properties['Pre_Select'] = Pre_Select

ch_diff = []
select_min_ind = 0
select_min = np.abs(np.sum(Pre_Select[:Ncoarse] - Centers))

for n in range(1, N_Pre - Ncoarse + 1):
    if np.abs(np.sum(Pre_Select[n:n + Ncoarse] - Centers)) < select_min:
        select_min_ind = n

properties['select_min_ind'] = select_min_ind

w = csv.writer(open('/Users/mike_e_dubs/python_stuff/MJW-MWA/MWA_Specs.csv', 'w'))

for key, val in properties.items():
    w.writerow([key, val])
