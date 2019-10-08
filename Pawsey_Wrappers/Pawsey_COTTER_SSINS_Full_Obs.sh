#!/bin/bash

module use /group/mwa/software/modulefiles
module load cotter
module load six
module load pyuvdata
module load h5py
module load scipy
module load matplotlib
module load numpy
module load pyyaml

data_dir=/astro/mwaeor/MWA/data

gpufiles=$(ls ${data_dir}/${obs}/*gpubox*.fits)
N_gpufiles=$(ls ${data_dir}/${obs}/*gpubox*.fits | wc -l)
mod=$((N_gpufiles % 24))
uvfits_dir=${data_dir}/${obs}/SSINS_uvfits

echo $obs

if [ ! -d $uvfits_dir ]
then
  mkdir $uvfits_dir
fi

echo $timeres

if [ ! -e ${uvfits_dir}/${obs}_noflag.uvfits ]; then
  echo $obs
  echo "Executing COTTER"
  gpufiles=$(ls ${data_dir}/${obs}/*gpubox*.fits)
  cotter -o ${uvfits_dir}/${obs}_noflag.uvfits -m ${data_dir}/${obs}/${obs}_metafits_ppds.fits -timeres $timeres -freqres $freqres -norfi -noflagautos -allowmissing -flagdcchannels -edgewidth $edgewidth -initflag $initflag -endflag $endflag -allowmissing $gpufiles
fi

echo "Executing python script for ${obs}"
python /home/mwilensky/MJW-MWA/Pawsey_Wrappers/SSINS_Gen.py $obs ${uvfits_dir}/${obs}_noflag.uvfits ${outdir}
rm -f ${uvfits_dir}/${obs}_noflag.uvfits
