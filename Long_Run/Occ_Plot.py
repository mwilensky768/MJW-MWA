import pickle

basedir = '/Users/mike_e_dubs/MWA/INS/Long_Run/Original_Jackknife'
occ_dict = pickle.load(open('%s/long_run_original_occ_dict.pik' % basedir, 'rb'))
slice_dict = pickle.load(open('%s/long_run_shape_dict.pik' % basedir, 'rb'))
