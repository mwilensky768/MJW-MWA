from SSINS import util
import numpy as np

bins = np.arange(-4, 5)
counts = np.array([1, 2, 5, 10, 10, 5, 2, 1])

stat, p = util.chisq(counts, bins, weight='exp', thresh=5)
print(counts, bins, stat, p)
