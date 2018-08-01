#!/bin/bash

####################################################
#
# RUN_FHD_AWS.SH
#
# Top level script to run a list of observation IDs through Mike Wilensky's
# RFI code on AWS.
#
#
# Required input arguments are obs_file_name (-f /path/to/obsfile)
#
# Optional input arguments are:
# starting_obs (-s 1061311664) which is defaulted to the beginning obsid of
# the specified file
# ending_obs (-e 1061323008) which is defaulted to the ending obsid of the
# specified file
# outdir (-o /path/to/output/directory) which is defaulted to /FHD_output
# nslots (-n 10) which is defaulted to 10
#
# This is adapted by R. Byrne from PIPE_DREAM.SH for running FHD on MIT
# (written by N. Barry)
# This was further adapted by Mike Wilensky
####################################################

#Clear input parameters
unset obs_file_name
unset starting_obs
unset ending_obs
unset outdir

#######Gathering the input arguments and applying defaults if necessary

#Parse flags for inputs
while getopts ":f:s:e:o:b:n:r:p:q:a:" option
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
    q) script=$OPTARG;; #The script to run
    a) set -f
       IFS=' '
       add_args=($OPTARG) ;; #Additional arguments for the script
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

#Find the max and min of the obs id array
max=${obs_id_array[0]}
min=${obs_id_array[0]}

for obs_id in "${obs_id_array[@]}"
do
   #Update max if applicable
   if [[ "$obs_id" -gt "$max" ]]
   then
	max="$obs_id"
   fi

   #Update min if applicable
   if [[ "$obs_id" -lt "$min" ]]
   then
	min="$obs_id"
   fi
done

#If minimum not specified, start at minimum of obs_file
if [ -z ${starting_obs} ]
then
   echo "Starting observation not specified: Starting at minimum of $obs_file_name"
   starting_obs=$min
fi

#If maximum not specified, end at maximum of obs_file
if [ -z ${ending_obs} ]
then
   echo "Ending observation not specified: Ending at maximum of $obs_file_name"
   ending_obs=$max
fi

#Create a list of observations using the specified range, or the full observation id file.
unset good_obs_list
for obs_id in "${obs_id_array[@]}"; do
    if [ $obs_id -ge $starting_obs ] && [ $obs_id -le $ending_obs ]; then
	good_obs_list+=($obs_id)
    fi
done

#######End of gathering the input arguments and applying defaults if necessary


#######Submit the firstpass jobs and wait for output

for obs_id in "${good_obs_list[@]}"
do
   qsub -V -b y -cwd -v nslots=${nslots},outdir=${outdir},s3_path=${s3_path},obs_id=$obs_id,uvfits_s3_loc=$uvfits_s3_loc,script=$script,add_args=$add_args -e ${logdir} -o ${logdir} -pe smp ${nslots} -sync y ~/MWA/MJW-MWA/rfi_job_aws.sh &
done
