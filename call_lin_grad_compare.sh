#!/bin/bash

for order in {0..5}
do
  python /lustre/aoc/projects/hera/mwilensk/MJW-MWA/MS_line_grad/INS_lin_grad_compare.py $basedir ${outdir}_${order} $obsfile $flag_choice $order
done
