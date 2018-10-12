#!/bin/bash

module use /group/mwa/software/modulefiles
module load six
module load pyuvdata/master
module load h5py/2.8.0
module load scipy
module load matplotlib
module load numpy/1.15.1
module load cotter
data_dir=/astro/mwaeor/MWA/data


# gpufiles=$(ls ${data_dir}/${obs}/*gpubox*)
if [ ! -e ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ]; then
  echo "Executing COTTER"
  cotter -o ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits -m ${data_dir}/${obs}/${obs}_metafits_ppds.fits -timeres 0.5 -freqres 40 -norfi -noflagdcchannels -edgewidth 0 -initflag 0 -allowmissing $gpufiles
fi

if [ ! -e ${outdir}_noavg/arrs/${obs}_None_INS_data.npym ]; then
  echo "Executing python script"
  python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_noavg
fi
