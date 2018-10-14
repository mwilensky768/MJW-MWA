#!/bin/bash

module load cotter
data_dir=/astro/mwaeor/MWA/data

if [ ! -e ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ]; then
  echo "Executing COTTER"
  gpufiles=$(ls ${data_dir}/${obs}/*gpubox*.fits)
  cotter -o ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits -m ${data_dir}/${obs}/${obs}_metafits_ppds.fits -timeres 0.5 -freqres 40 -norfi -noflagdcchannels -edgewidth 0 -initflag 0 $gpufiles
fi
