#!/bin/bash

for list in $(ls .)
do
  day=${list:2:7}
  for i in {1..5}
  do
    obsid=$(head -"$i" $file | tail -1)
    outdir=/lustre/aoc/projects/hera/mwilensk/HERA_IDR2_Prelim_Set
    for pol in xx yy xy yx
    do
      obs=${obsid}.${pol}.HH
      indir=/lustre/aoc/projects/hera/H1C_IDR2/${day}/${obs}.uv
      python /lustre/aoc/projects/hera/mwilensk/SSINS/Scripts/Catalog_Gen.py $obs $indir $outdir
    done
  done
done
