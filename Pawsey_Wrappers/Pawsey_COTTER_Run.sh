#!/bin/bash

module use /group/mwa/software/modulefiles
module load cotter
data_dir=/astro/mwaeor/MWA/data

if [ ! -e ${data_dir}/${obs}/${obs}_noflag_box1_noavg.uvfits ]; then
  echo $obs
  echo "Executing COTTER"
  gpufiles=$(ls ${data_dir}/${obs}/*gpubox01*.fits)
  cotter -o ${data_dir}/${obs}/${obs}_noflag_box1_noavg.uvfits -m ${data_dir}/${obs}/${obs}_metafits_ppds.fits -timeres 0.5 -freqres 40 -norfi -noflagdcchannels -edgewidth 0 -initflag 0 -endflag 0 -allowmissing -sbcount 1 $gpufiles
fi
