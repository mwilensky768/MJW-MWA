#!/bin/bash

basedir=/lustre/aoc/projects/hera/mwilensk/HERA_IDR2_Prelim_Set_nocut
outdir=${basedir}_compare
obsfile=/lustre/aoc/projects/hera/mwilensk/HERA_IDR2_Prelim_obs.txt
flag_choice=None

for order in {0..5}
do
  python /lustre/aoc/projects/hera/mwilensk/MJW-MWA/MS_line_grad/INS_lin_grad_compare.py $basedir ${outdir}_${order} $obsfile $flag_choice $order
done
