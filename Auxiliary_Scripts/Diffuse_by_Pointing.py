import numpy as np
import csv

dir = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Obs_Lists'

with open('%s/Diffuse_by_pointing.txt' % (dir)) as f:
    lst = list(csv.reader(f))

lst = [[line[0], (int(line[1][2:]), int(line[2][:-1]))] for line in lst]
points = list(set([line[1] for line in lst]))

for point in points:
    sublst = [line[0] for line in lst if line[1] == point]
    file = open('%s/diffuse_point_%i_%i.txt' % (dir, point[0], point[1]), 'w')
    for obs in sublst:
        file.write('%s\n' % (obs))
