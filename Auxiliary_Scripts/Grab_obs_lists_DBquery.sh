#!/bin/bash

while getopts ":n:p:m:x:" option
do
  case $option in
    n) n_pages=$OPTARG;; # How many pages to iterate over
    p) pointing=$OPTARG;; # which pointing is it?
    m) mindate=$OPTARG;; # The mintime (UTC)
    x) maxdate=$OPTARG;; # The maxtime (UTC)
  esac
done

#Manual shift to the next flag.
shift $(($OPTIND - 1))

case $pointing in

  minus_two)
    gridpoint=3
    ;;

  minus_one)
    gridpoint=1
    ;;

  zenith)
    gridpoint=0
    ;;

  plus_one)
    gridpoint=2
    ;;

  plus_two)
    gridpoint=4
    ;;

esac

echo gridpoint is $gridpoint

for ra_max in 360 10; do
  if [ ${ra_max} == 360 ]; then
    ra_min=350
  else
    ra_min=0
  fi
  echo "ra_min is ${ra_min}"
  for page in $(seq 1 $n_pages); do
    wget "http://ws.mwatelescope.org/metadata/find?projectid=G0009&mintime_utc=${mindate}&maxtime_utc=${maxdate}&minduration=112&minra=${ra_min}&maxra=${ra_max}&mindec=-37&maxdec=-17&gridpoint=${gridpoint}&anychan=140&page=${page}&pagesize=500&minfiles=24" -O 2014_EoR_High_RA_${ra_min}_pointing_${pointing}_page_${page}.json
    sleep 120
  done
done
