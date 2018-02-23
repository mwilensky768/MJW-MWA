#!/bin/bash

obs_file_name='/nfs/eor-00/h1/mwilensk/PPD/PPD_Obs.txt'
obs_path_file_name='/nfs/eor-00/h1/mwilensk/PPD/PPD_Obs_paths.txt'
N_lines=$(wc -l ${obs_file_name})

for i in {1..16}
do
	obs=$(sed -n "${i}p" ${obs_file_name})
	obs_path="$(python /nfs/eor-00/h1/mwilensk/MWA_Tools/scripts/read_uvfits_loc.py -v 5 -s 1 -o ${obs})"
	if [[-z "$obs_path"]]
	then
		obs_path="$(python /nfs/eor-00/h1/mwilensk/MWA_Tools/scripts/read_uvfits_loc.py -v 4 -s 1 -o ${obs})"
	fi
	echo "${obs_path}" >> $obs_path_file_name
done
