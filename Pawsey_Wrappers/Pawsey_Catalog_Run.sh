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

python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits $outdir_0_112 --time_range 112 224
python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits $outdir_112_224 --time_range 0 112
