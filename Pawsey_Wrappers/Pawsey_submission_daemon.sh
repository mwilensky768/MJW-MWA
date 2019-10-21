#!/bin/bash

# Run this shit from tmux

wrapper=/home/mwilensky/MJW-MWA/Pawsey_Wrappers/Pawsey_COTTER_SSINS_Wrapper.sh
2014_SSINS_dir=/group/mwaeor/mwilensky/2014_SSINS_outputs
script=/home/mwilensky/MJW-MWA/Pawsey_Wrappers/Pawsey_COTTER_SSINS_Full_Obs.sh

for pointing in minus_two minus_one zenith plus_one plus_two; do
  bash $wrapper -f ${SSINS_dir}/2014_EoR_High_Pointing_${pointing}_obs.txt -o ${SSINS_dir}/2014_${pointing}_outputs -s $script -m 50G -t 02:00:00 -a 2 -b 80 -e 80 -i 2 -c 2
  sleep 2h
done
