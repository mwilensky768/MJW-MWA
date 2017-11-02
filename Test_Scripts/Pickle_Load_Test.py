import matplotlib.pyplot as plt
import pickle

fig = pickle.load(file('/Users/mike_e_dubs/MWA/Misc/Pickle_Test_Fig.p', 'r'))
ax = pickle.load(file('/Users/mike_e_dubs/MWA/Misc/Pickle_Test_Ax.p', 'r'))

plt.show()
