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


if [ ! -e ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ]; then
  echo "Executing COTTER"
  gpufiles=$(ls ${data_dir}/${obs}/*gpubox*)
  cotter -o ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits -m ${data_dir}/${obs}/${obs}_metafits_ppds.fits -timeres 0.5 -freqres 40 -norfi -noflagdcchannels -edgewidth 0 -initflag 0 -allowmissing $gpufiles
fi

if [ ! -e ${outdir}_noavg/arrs/${obs}_None_INS_data.npym ]; then
  echo "Executing python script"
  echo $obs
  python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_noavg_0_384 --freq_range 0 384
  python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_noavg_384_768 --freq_range 384 768
fi
