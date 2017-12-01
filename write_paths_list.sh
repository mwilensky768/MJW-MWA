#!/bin/bash

obs_file_name='/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/S2_Zenith_Calcut_Autos_Rogue_OBSIDS.txt'
obs_path_file_name='/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/S2_Zenith_Calcut_Autos_Rogue_OBSIDS_paths.txt'
N_lines=$(wc -l ${obs_file_name})

for i in {1..1}
do
	obs=$(sed -n "${i}p" ${obs_file_name})
	obs_path="$(python /nfs/eor-00/h1/mwilensk/MWA_Tools/scripts/read_uvfits_loc.py -v 5 -s 1 -o ${obs})"
	echo "${obs_path}" >> $obs_path_file_name
done
