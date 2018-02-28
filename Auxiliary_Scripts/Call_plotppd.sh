#!/bin/bash

obs_file_name='/nfs/eor-00/h1/mwilensk/PPD/PPD_Obs.txt'

for i in {1..29}
do
  obs=$(sed -n "${i}p" ${obs_file_name})
  python /nfs/eor-00/h1/mwilensk/MWA_Tools/scripts/plotppds.py /nfs/eor-14/r1/EoRuvfits/PPD_clip_metafits/${obs}/${obs}_metafits_ppds.fits -g -f /nfs/eor-00/h1/mwilensk/PPD/${obs}.png
done
