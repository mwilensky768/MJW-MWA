import pickle
import rfipy as rfi
import matplotlib.pyplot as plt

RFI = rfi.RFI('s1061313008', '/Users/mike_e_dubs/MWA/Data/smaller_uvfits/s1061313008.uvfits')

hist = RFI.one_d_hist_prepare()

fig, ax = plt.subplots()

RFI.one_d_hist_plot(fig, ax, hist, 's1061313008 Pickle Test')

fig.savefig('/Users/mike_e_dubs/MWA/Misc/Pickle_Test.png')

pickle.dump(ax, file('/Users/mike_e_dubs/MWA/Misc/Pickle_Test_Ax.p', 'w'))
pickle.dump(fig, file('/Users/mike_e_dubs/MWA/Misc/Pickle_Test_Fig.p', 'w'))
