#!/bin/bash

while getopts ":n:p:g:" option
do
  case $option in
    n) Npage=$OPTARG;; # Number of pages to go through
    p) pointing=$OPTARG;; # which pointing is it?
    g) gridpoint=$OPTARG;; # The gridpoint of the obs
  esac
done

#Manual shift to the next flag.
shift $(($OPTIND - 1))

for page_num in {1..3};
do
  echo page_num $page_num
  #wget "http://ws.mwatelescope.org/metadata/find?mintime=1252195218&maxtime=1255219218&extended=1&page=$i"
  wget "http://ws.mwatelescope.org/metadata/find?projectid=G0009&creator=DJacobs&mintime_utc=2014-07-01T00%3A00%3A00.000&maxtime_utc=2015-01-01T00%3A00%3A00.000&minduration=112&minra=-10&maxra=10&mindec=-37&maxdec=-17&gridpoint=${gridpoint}&anychan=140&page=${page_num}" -O 2014_EoR_High_pointing_${pointing}_obs_page_${page_num}.json
  sleep 77
done
