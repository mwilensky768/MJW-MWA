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

python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_0_28_noavg --time_range 0 28
python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_28_56_noavg --time_range 28 56
python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_56_84_noavg --time_range 56 84
python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_84_112_noavg --time_range 84 112
python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_112_140_noavg --time_range 112 140
python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_140_168_noavg --time_range 140 168
python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_168_196_noavg --time_range 168 196
python /home/mwilensky/MJW-MWA/Catalog_Gen.py $obs ${data_dir}/${obs}/${obs}_noavg_noflag.uvfits ${outdir}_196_224_noavg --time_range 196_224
