#!/bin/bash

module use /group/mwa/software/modulefiles
module load cotter
module load six
module load pyuvdata/master
module load h5py
module load scipy
module load matplotlib
module load numpy

data_dir=/astro/mwaeor/MWA/data

gpufiles=$(ls ${data_dir}/${obs}/*gpubox*.fits)
N_gpufiles=$(ls ${data_dir}/${obs}/*gpubox*.fits | wc -l)
mod=$((N_gpufiles % 24))
uvfits_dir=${data_dir}/${obs}/SSINS_uvfits
meta_override_dir=/group/mwaeor/mwilensky/meta_override

echo $obs

if [ ! -d $uvfits_dir ]
then
  mkdir $uvfits_dir
fi

if [ mod -eq 0 ]; then
  echo "All gpubox files are present"
  for i in {1..24}
  do
    if [ i -lt 10]; then
      j=0$i
    else
      j=$i
    fi
    gpufiles=$(ls ${data_dir}/${obs}/*gpubox${j}*.fits)
    echo "Executing COTTER for ${obs}"
    cotter -o ${uvfits_dir}/${obs}_box${j}.uvfits -m ${data_dir}/${obs}/${obs}_metafits_ppds.fits -timeres $timeres -freqres $freqres -norfi -noflagdcchannels -edgewidth $edgewidth -initflag $initflag -endflag $endflag -allowmissing -sbstart $i -sbcount 1 -h ${meta_override_dir}/meta_override_${i}.txt $gpufiles
    echo "Executing python script for ${obs}"
    python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${uvfits_dir}/${obs}_box${j}.uvfits ${outdir}_box${i}
    rm -f ${uvfits_dir}/${obs}_box${j}.uvfits
  done
echo "Concatenating Incoherent Noise Spectra"
python /home/mwilensky/MJW-MWA/INS_Concatenate.py $obs $outdir
else
  echo "Not all gpubox files are present. Not computing spectrum."
