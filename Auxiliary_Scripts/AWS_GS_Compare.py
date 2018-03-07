gs_list_file = '/Users/mike_e_dubs/MWA/Obs_Lists/Golden_Set_OBSIDS.txt'
aws_list_file = '/Users/mike_e_dubs/MWA/Obs_Lists/obs_on_aws.txt'

with open(gs_list_file) as f:
    gs_list = f.read().split("\n")
with open(aws_list_file) as g:
    aws_list = g.read().split("\n")

gs_list.remove('')
aws_list.remove('')

intersect_list = [line[-17:-7] for line in aws_list if line[-17:-7] in gs_list]

fil = open('/Users/mike_e_dubs/MWA/Obs_Lists/aws_gs_intersect.txt', 'w')

for obs in intersect_list:
    fil.write('%s\n' % (obs))
