#!/bin/bash

for plot in $(ls /lustre/aoc/projects/hera/mwilensk/HERA_IDR2_Prelim_Set_nocut/figs/*data.png)
do
  echo ${plot:67:23} >> /lustre/aoc/projects/hera/mwilensk/HERA_IDR2_Prelim_obs.txt
done
