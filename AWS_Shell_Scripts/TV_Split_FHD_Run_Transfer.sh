#!/bin/bash

####################################################
#
# RUN_FHD_AWS.SH
#
# Top level script to run a list of observation IDs through FHD (deconvolution
# or firstpass) on AWS.
#
# Required input arguments are obs_file_name (-f /path/to/obsfile) and version
# (-v yourinitials_jackknife_test)
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
# Example command: bash run_fhd_aws.sh -f ~/MWA/MJW-MWA/Obs_Lists/GS_NB.txt -o /NB_Image -b s3://mw-mwa-ultra-faint-rfi -v mjw_vanilla_test -u mjw

####################################################

#Clear input parameters
unset obs_file_name
unset starting_obs
unset ending_obs
unset outdir
unset version

#######Gathering the input arguments and applying defaults if necessary

#Parse flags for inputs
while getopts ":f:s:e:o:b:v:n:r:u:p:m:i:j:c:" option
do
   case $option in
    f) obs_file_name="$OPTARG";;	#text file of observation id's
    s) starting_obs=$OPTARG;;	#starting observation in text file for choosing a range
    e) ending_obs=$OPTARG;;		#ending observation in text file for choosing a range
    o) outdir=$OPTARG;;		#output directory for FHD
    b) s3_path=$OPTARG;;		#output bucket on S3
    v) version=$OPTARG;;		#FHD folder name and case
		#Example: nb_foo creates folder named fhd_nb_foo
    n) nslots=$OPTARG;;		#Number of slots for grid engine
    u) user=$OPTARG;;		#User: options are rlb (default) or nb
    p) uvfits_s3_loc=$OPTARG;;		#Path to uvfits files on S3
    m) metafits_s3_loc=$OPTARG;;		#Path to metafits files on S3
    i) input_vis=$OPTARG;;              #Optional input visibilities for in situ sim
    j) input_eor=$OPTARG;;             #Optional input eor sim for in situ sim
    c) cal_s3_loc=$OPTARG;; #Path to calibration on s3
    T) TV_min=$OPTARG;; #text file of tv min
    V) TV_max=$OPTARG;; #text file of tv max
    C) cal_min=$OPTARG;; #text file of cal min
    A) cal_max=$OPTARG;; #text file of cal max
    H) chan=$OPTARG;; #text file of chan

    \?) echo "Unknown option: Accepted flags are -f (obs_file_name), -s (starting_obs), -e (ending obs), -o (output directory), "
        echo "-b (output bucket on S3), -v (version input for FHD),  -n (number of slots to use), "
        echo "-u (user), -p (path to uvfits files on S3), -m (path to metafits files on S3)."
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
    outdir=/FHD_output
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

if [ -z ${metafits_s3_loc} ]; then
    metafits_s3_loc=s3://mwapublic/metafits/4.1
else
    #strip the last / if present in metafits filepath
    metafits_s3_loc=${metafits_s3_loc%/}
fi

if [ -z ${s3_path} ]
then
    s3_path=s3://mwatest/diffuse_survey
    echo Using default S3 location: $s3_path
else
    #strip the last / if present in output directory filepath
    s3_path=${s3_path%/}
    echo Using S3 bucket: $s3_path
fi

logdir=~/grid_out

if [ -z ${version} ]; then
   echo Please specify a version, e.g, yourinitials_test
   exit 1
fi

if [ -z ${user} ]; then
    user='rlb'
fi
if [ ${user} != 'rlb' ] && [ ${user} != 'nb' ] && [ ${user} != 'mjw' ]; then
    echo Invalid user. Option are -u rlb and -u nb and -u mjw
    exit 1
fi

if [ $user == 'nb' ]; then
    versions_script='nb_eor_firstpass_versions'
elif [ $user == 'rlb' ]; then
    versions_script='fhd_versions_rlb'
elif [ $user == 'mjw' ]; then
    versions_script='mjw_fhd_versions'
fi

#Set typical slots needed for standard FHD firstpass if not set.
if [ -z ${nslots} ]; then
    nslots=10
fi



#Make directory if it doesn't already exist
sudo mkdir -p -m 777 ${outdir}/fhd_${version}/grid_out
echo Output located at ${outdir}/fhd_${version}

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

#######End of gathering the input arguments and applying defaults if necessary


#######Submit the firstpass jobs and wait for output

for i in {0..22}
do
   qsub -V -b y -cwd -v nslots=${nslots},outdir=${outdir},version=${version},s3_path=${s3_path},obs_id=$obs_id_array[${i}],versions_script=$versions_script,uvfits_s3_loc=$uvfits_s3_loc,metafits_s3_loc=$metafits_s3_loc,input_vis=$input_vis,input_eor=$input_eor,cal_s3_loc=$cal_s3_loc,TV_min=${TV_min_array[${i}]},TV_max=${TV_max_array[${i}]},cal_min=${cal_min_array[${i}]},cal_max=${cal_max_array[${i}]},chan=${chan_array[${i}]} -e ${logdir} -o ${logdir} -pe smp ${nslots} -sync y ~/MWA/MJW-MWA/AWS_Shell_Scripts/TV_Split_FHD_Job_Transfer.sh &
done
