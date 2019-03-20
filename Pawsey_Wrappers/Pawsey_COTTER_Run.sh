#!/bin/bash

module use /group/mwa/software/modulefiles
module load cotter
data_dir=/astro/mwaeor/MWA/data

# Set default timeres
if [ -z ${timeres} ]; then
  timeres=2
fi

# Set default freqres
if [ -z ${freqres} ]; then
  freqres=80
fi

# Set default edgewidth
if [ -z ${edgewidth} ]; then
  edgewidth=80
fi

# Set default initflag
if [ -z ${initflag} ]; then
  initflag=2
fi

# Set default endflag
if [ -z ${endflag} ]; then
  endflag=2
fi

if [ ! -e ${data_dir}/${obs}/${obs}_noflag.uvfits ]; then
  echo $obs
  echo "Executing COTTER"
  gpufiles=$(ls ${data_dir}/${obs}/*gpubox*.fits)
  cotter -o ${data_dir}/${obs}/${obs}_noflag.uvfits -m ${data_dir}/${obs}/${obs}_metafits_ppds.fits -timeres $timeres -freqres $freqres -norfi -noflagautos -allowmissing -flagdcchannels -edgewidth $edgewidth -initflag $initflag -endflag $endflag -allowmissing $gpufiles
fi
