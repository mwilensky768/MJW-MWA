#!/bin/bash

module use /group/mwa/software/modulefiles
module load numpy
module load astropy

python ~/MJW-MWA/1061313128/1061313128_mwaf_map.py
