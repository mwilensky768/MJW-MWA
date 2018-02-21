#!/bin/bash

obs_file_name='/nfs/eor-00/h1/mwilensk/P2_Bad_Obs/badobs_list_wenyang.txt'

for i in {2..16}
do
  obs=$(sed -n "${i}p" ${obs_file_name})
  python /nfs/eor-00/h1/mwilensk/MWA_Tools/scripts/obsdownload.py -u -o ${obs}
done
