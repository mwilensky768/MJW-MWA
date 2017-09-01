#!/bin/bash

obs_file_name='/nfs/eor-00/h1/mwilensk/FHD/obs_list/beardsley_thesis_list.txt'
obs_path_file_name='/nfs/eor-00/h1/mwilensk/FHD/obs_list/beardsley_thesis_list_paths.txt'

for i in {1..1029}
do
	obs=$(sed -n "${i}p" ${obs_file_name})
	obs_path="$(python /nfs/eor-00/h1/mwilensk/MWA_Tools/scripts/read_uvfits_loc.py -v 4 -s 1 -o ${obs})"
	echo "${obs_path}" >> $obs_path_file_name
done
