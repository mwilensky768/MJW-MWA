#!/bin/bash

#Clear input parameters
unset obs_file_name
unset starting_obs
unset ending_obs
unset outdir

#Parse flags for inputs
while getopts ":f:s:e:o:b:n:r:p:T:V:c:a:h:" option
do
   case $option in
    f) obs_file_name="$OPTARG";;	#text file of observation id's
    s) starting_obs=$OPTARG;;	#starting observation in text file for choosing a range
    e) ending_obs=$OPTARG;;		#ending observation in text file for choosing a range
    o) outdir=$OPTARG;;		#output directory for FHD
    b) s3_path=$OPTARG;;		#output bucket on S3
		#Example: nb_foo creates folder named fhd_nb_foo
    n) nslots=$OPTARG;;		#Number of slots for grid engine
    p) uvfits_s3_loc=$OPTARG;;		#Path to uvfits files on S3
    T) TV_min=$OPTARG;; #text file of tv min
    V) TV_max=$OPTARG;; #text file of tv max
    c) cal_min=$OPTARG;; #text file of cal min
    a) cal_max=$OPTARG;; #text file of cal max
    h) chan=$OPTARG;; #text file of chan
    \?) echo "Unknown option: Accepted flags are -f (obs_file_name), -s (starting_obs), -e (ending obs), -o (output directory), "
        echo "-b (output bucket on S3),  -n (number of slots to use), "
        echo "-u (user), -p (path to uvfits files on S3)."
        exit 1;;
    :) echo "Missing option argument for input flag"
       exit 1;;
   esac
done

#Manual shift to the next flag.
shift $(($OPTIND - 1))

#Throw error if no obs_id file.
if [ -z ${obs_file_name} ]; then
   echo "Need to specify a full filepath to a list of viable observation ids."
   exit 1
fi

#Update the user on which obsids will run given the inputs
if [ -z ${starting_obs} ]
then
    echo Starting at observation at beginning of file $obs_file_name
else
    echo Starting on observation $starting_obs
fi

if [ -z ${ending_obs} ]
then
    echo Ending at observation at end of file $obs_file_name
else
    echo Ending on observation $ending_obs
fi

#Set default output directory if one is not supplied and update user
if [ -z ${outdir} ]
then
    outdir=/rfi_output
    echo Using default output directory: $outdir
else
    #strip the last / if present in output directory filepath
    outdir=${outdir%/}
    echo Using output directory: $outdir
fi

if [ -z ${uvfits_s3_loc} ]; then
    uvfits_s3_loc=s3://mwapublic/uvfits/4.1
else
    #strip the last / if present in uvfits filepath
    uvfits_s3_loc=${uvfits_s3_loc%/}
fi

if [ -z ${s3_path} ]
then
    s3_path=s3://mwa-data/golden_set_rfi
    echo Using default S3 location: $s3_path
else
    #strip the last / if present in output directory filepath
    s3_path=${s3_path%/}
    echo Using S3 bucket: $s3_path
fi

logdir=~/grid_out

#Set typical slots needed for standard FHD firstpass if not set.
if [ -z ${nslots} ]; then
    nslots=10
fi



#Make directory if it doesn't already exist
sudo mkdir -p -m 777 ${outdir}/grid_out
echo Output located at ${outdir}

#Read the obs file and put into an array, skipping blank lines if they exist
i=0
while read line
do
   if [ ! -z "$line" ]; then
      obs_id_array[$i]=$line
      i=$((i + 1))
   fi
done < "$obs_file_name"

#Read the TV_mins and put into an array, skipping blank lines if they exist
i=0
while read line
do
   if [ ! -z "$line" ]; then
      TV_min_array[$i]=$line
      i=$((i + 1))
   fi
done < "$TV_min"

#Read the TV_maxs and put into an array, skipping blank lines if they exist
i=0
while read line
do
   if [ ! -z "$line" ]; then
      TV_max_array[$i]=$line
      i=$((i + 1))
   fi
done < "$TV_max"

#Read the cal_mins and put into an array, skipping blank lines if they exist
i=0
while read line
do
   if [ ! -z "$line" ]; then
      cal_min_array[$i]=$line
      i=$((i + 1))
   fi
done < "$cal_min"

#Read the cal_maxs and put into an array, skipping blank lines if they exist
i=0
while read line
do
   if [ ! -z "$line" ]; then
      cal_max_array[$i]=$line
      i=$((i + 1))
   fi
done < "$cal_max"

#Read the chans and put into an array, skipping blank lines if they exist
i=0
while read line
do
   if [ ! -z "$line" ]; then
      chan_array[$i]=$line
      i=$((i + 1))
   fi
done < "$chan"

for i in "${1..23}"
do
   qsub -V -b y -cwd -v nslots=${nslots},outdir=${outdir},s3_path=${s3_path},obs_id=${obs_id_array[${i}]},uvfits_s3_loc=$uvfits_s3_loc,TV_min=${TV_min_array[${i}]},TV_max=${TV_max_array[${i}]},cal_min=${cal_min_array[${i}]},cal_max=${cal_max_array[${i}]},chan=${chan_array[${i}]} -e ${logdir} -o ${logdir} -pe smp ${nslots} -sync y ~/MWA/MJW-MWA/AWS_Shell_Scripts/TV_Split_Job_AWS.sh &
done
