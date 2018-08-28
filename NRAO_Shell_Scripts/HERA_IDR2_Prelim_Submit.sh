#!/bin/bash

for list in $(ls /lustre/aoc/projects/hera/mwilensk/Obs_Lists/Obs_Select)
do
  day=${list:0:7}
  for i in {1..5}
  do
    obsid=$(head -"$i" $list | tail -1)
    echo $obsid
    outdir=/lustre/aoc/projects/hera/mwilensk/HERA_IDR2_Prelim_Set
    for pol in xx yy xy yx
    do
      obs=${obsid}.${pol}.HH
      indir=/lustre/aoc/projects/hera/H1C_IDR2/${day}/${obs}.uv
      python /lustre/aoc/projects/hera/mwilensk/MJW-MWA/Catalog_Gen.py $obs $indir $outdir
    done
  done
done
