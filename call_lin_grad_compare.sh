#!/bin/bash

for order in {0..5}
do
  /lustre/aoc/projects/hera/mwilensk/MJW-MWA/MS_line_grad/INS_lin_grad_compare.py $basedir $outdir $obsfile $flag_choice $order
done
