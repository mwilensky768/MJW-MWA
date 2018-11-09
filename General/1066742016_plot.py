from SSINS import INS, plot_lib, util
import matplotlib.pyplot as plt
import numpy as np
import os

obs = '1066742016'
indir = '/Users/mike_e_dubs/MWA/INS/Long_Run/All'
outpath = '/Users/mike_e_dubs/General/1066742016'
if not os.path.exists(outpath):
    os.makedirs(outpath)

read_paths = util.read_paths_construct(indir, None, obs, 'INS')
ins = INS(read_paths=read_paths, outpath=outpath, obs=obs)
fig, ax = plt.subplots(figsize=(8, 9))
fig_diff, ax_diff = plt.subplots(figsize=(8, 9))
