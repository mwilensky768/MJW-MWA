#!/bin/bash

module use /group/mwa/software/modulefiles
module load cotter
data_dir=/astro/mwaeor/MWA/data

if [ ! -e ${data_dir}/${obs}/${obs}_noflag.uvfits ]; then
  echo $obs
  echo "Executing COTTER"
  gpufiles=$(ls ${data_dir}/${obs}/*gpubox*.fits)
  cotter -o ${data_dir}/${obs}/${obs}_noflag.uvfits -m ${data_dir}/${obs}/${obs}_metafits_ppds.fits -timeres 2 -freqres 80 -norfi -noflagdcchannels -edgewidth 80 -initflag 2 -endflag 6 -allowmissing $gpufiles
fi
