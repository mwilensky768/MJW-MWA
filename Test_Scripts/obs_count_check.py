import numpy as np
import matplotlib.pyplot as plt

obs_count = np.load('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_Temp_Obs_Count.npy')

plt.plot(range(len(obs_count)), obs_count)
plt.show()
