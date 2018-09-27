#!/bin/bash

module use /group/mwa/software/modulefiles
module load numpy/1.15.1
module load scipy/1.1.0
module load matplotlib/2.2.3
module load h5py/2.8.0
module load six
module load pyuvdata/master
module load cotter
data_dir=/astro/mwaeor/MWA/data


# gpufiles=$(ls ${data_dir}/${obs}/*gpubox*)
# cotter -o ${data_dir}/${obs}/${obs}_80khz_noflag.uvfits -m ${data_dir}/${obs}/${obs}_metafits_ppds.fits -timeres 0.5 -freqres 80 -norfi -noflagdcchannels -edgewidth 0 -initflag 0 $gpufiles

python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_0_56 --time_range 0 56
python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_56_112 --time_range 56 112
python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_112_168 --time_range 112 168
python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_168_224 --time_range 168 224
