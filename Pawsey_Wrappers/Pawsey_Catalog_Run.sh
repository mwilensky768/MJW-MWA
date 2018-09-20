#!/bin/bash

gpufiles=$(ls ${data_dir}/${obs}/*gpubox*)
cotter -o ${datadir}/${obs}_noavg_noflag.uvfits -m ${data_dir}/${obs}/${obs}_metafits_ppds.fits -timeres 0.5 -freqres 40 -norfi -noflagdcchannels -edgewidth 0 -initflag 0 $gpufiles

python /home/mwilensk/MJW-MWA/Catalog_Run.py $obs ${data_dir}/${obs}_noavg_noflag.uvfits $outdir
