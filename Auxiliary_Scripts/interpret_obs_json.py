import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filepath_in')
parser.add_argument('filepath_out')
args = parser.parse_args()

with open(args.filepath_in, 'r') as json_file:
    obj = json.load(json_file)
    obslist = []
    for obs_ind in range(len(obj)):
        obsid = int(obj[obs_ind][0])
        obslist.append(obsid)

with open(args.filepath_out, 'w') as txt_file:
    for obsid in obslist:
        txt_file.write("%s\n" % obsid)
