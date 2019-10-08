import xmltodict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--xml_file_path', action='store', help='Path to the xml file from the ASVO query')
parser.add_argument('-o', '--outfile', action='store', help='Name of output file')
args = parser.parse_args()

# Makes a nested dictionary according to how the xml file is organized
with open(args.xml_file_path) as xml_file:
    dat_dict = xmltodict.parse(xml_file.read())

# Grab the relevant part of the dictionary (just the xml organization here)
table_dict = dat_dict['VOTABLE']['RESOURCE']['TABLE']['DATA']['TABLEDATA']['TR']

# Make a list of obs_ids (initialize as empty)
obs_list = []

for obs_dat in table_dict:
    # happens to be the obs_id
    obs_list.append(obs_dat['TD'][4])

with open(args.outfile, 'w') as obs_file:
    for obs in obs_list:
        obs_file.write("%s\n" % obs)
