#!/bin/bash

obs_file_name='/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Diffuse_2015/sidelobe_survey_obsIDs.txt'
obs_path_file_name='/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Diffuse_2015/sidelobe_survey_obsIDs_paths.txt'

for i in {1..934}
do
	obs=$(sed -n "${i}p" ${obs_file_name})
	obs_path="$(python /nfs/eor-00/h1/mwilensk/MWA_Tools/scripts/read_uvfits_loc.py -v 5 -s 1 -o ${obs})"
	echo "${obs_path}" >> $obs_path_file_name
done
