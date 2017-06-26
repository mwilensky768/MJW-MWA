import pyuvdata as pyuv
import numpy as np

UV = pyuv.UVData()
UV.read_uvfits('/nfs/mwa-14/r1/EoRuvfits/jd2456528v4_1/1061313128.uvfits')
UV.select(ant_pairs_nums = [UV.baseline_to_antnums(k) for k in (range(3)+np.ones(3))])
