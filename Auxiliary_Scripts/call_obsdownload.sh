#!/bin/bash


while read obs
do
  python /Users/mike_e_dubs/python_stuff/MWA_Tools/scripts/obsdownload.py -m -o ${obs}
done < ${1}
