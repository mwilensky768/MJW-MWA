from SSINS import SS

inpath = '/Users/mike_e_dubs/MWA/Data/uvfits/1118096056.uvfits'
outpath = '/Users/mike_e_dubs/MWA/1118096056'
obs = '1118096056'

ss = SS(obs=obs, inpath=inpath, outpath=outpath,
        read_kwargs={'ant_str': 'cross'}, flag_choice='original')
ss.INS_prepare()
ss.INS.save()
