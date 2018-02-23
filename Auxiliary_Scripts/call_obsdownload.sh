#!/bin/bash

obs_file_name='/nfs/eor-00/h1/mwilensk/PPD/PPD_Obs.txt'

for i in {1..29}
do
  obs=$(sed -n "${i}p" ${obs_file_name})
  python /nfs/eor-00/h1/mwilensk/MWA_Tools/scripts/obsdownload.py -m -o ${obs}
done
