from SSINS import SS
import os
import psutil

process = psutil.Process(os.getpid())


inpath = '/Users/mike_e_dubs/MWA/Data/uvfits/1061313128.uvfits'
print(process.memory_info().rss)
ss = SS(inpath=inpath, read_kwargs={'ant_str': 'cross'})
print(process.memory_info().rss)
ss.INS_prepare()
print(process.memory_info().rss)
