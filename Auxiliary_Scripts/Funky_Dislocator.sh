#!/bin/bash

obs_file_name='/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_8s_Autos_Funky.txt'
plots_path='/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_8s_Autos_Waterfall_Plots/'

for i in {1..36}
do
  obs=$(sed -n "${i}p" ${obs_file_name})
  cp ${plots_path}/${obs}_RFI_Diagnostic_All.png ${plots_path}/Funky/
done
