#!/bin/bash

module use /group/mwa/software/modulefiles
module load six
module load pyuvdata/master
module load h5py/2.8.0
module load scipy
module load matplotlib
module load numpy/1.15.1
data_dir=/astro/mwaeor/MWA/data

if [ ! -e ${outdir}_noavg/arrs/${obs}_None_INS_data.npym ]; then
  echo $obs
  if [ ! -e ${outdir}_noavg_0_384/arrs/${obs}_None_INS_data.npym ]; then
    echo "Executing first python script for ${obs}"
    python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_noavg_0_384 --freq_range 0 384
  fi
  if [ ! -e ${outdir}_noavg_384_768/arrs/${obs}_None_INS_data.npym ]; then
    echo "Executing second python script for ${obs}"
    python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_noavg_384_768 --freq_range 384 768
  fi
fi
