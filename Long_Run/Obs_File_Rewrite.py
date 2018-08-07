dir = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Obs_Lists/Long_Run/TV_Split'

for mode in ['Cal', 'TV']:
    f_list = []
    for fname in ['ObsList', '%s_Min' % mode, '%s_Max' % mode, 'Channel']:
        with open('%s/TV_Split_%s.txt' % (dir, fname)) as f:
            f_list.append(f.read().split("\n"))

    lines = []
    for i in range(len(f_list[0])):
        lines.append('%s_%s%s_t%s_t%s' % (f_list[0][i], mode.lower(),
                                          f_list[3][i], f_list[1][i],
                                          f_list[2][i]))

    F = open('%s/TV_Split_%s_Obslist.txt' % (dir, mode), 'w')
    for line in lines:
        F.write('%s\n' % line)
