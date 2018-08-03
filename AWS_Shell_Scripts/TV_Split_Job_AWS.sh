#!/bin/bash

#############################################################################
# Runs one observation at a time in grid engine.  Second level program for
# running rfi code on AWS machines. First level program is run_fhd_aws.sh
# This is a modified version of fhd_job_aws.sh written by Ruby Byrne
#############################################################################

echo JOBID ${JOB_ID}
echo TASKID ${SGE_TASK_ID}
echo OBSID ${obs_id}
echo "JOB START TIME" `date +"%Y-%m-%d_%H:%M:%S"`
myip="$(dig +short myip.opendns.com @resolver1.opendns.com)"
echo PUBLIC IP ${myip}

#set defaults
if [ -z ${nslots} ]; then
    nslots=10
fi
if [ -z ${outdir} ]; then
    outdir=/rfi_output/${obs_id}_rfi
fi
if [ -z ${s3_path} ]; then
    s3_path=s3://mwa-data/golden_set_rfi
fi
if [ -z ${uvfits_s3_loc} ]; then
    uvfits_s3_loc=s3://mwapublic/uvfits/4.1
fi
if [ -z ${obs_id} ]; then
    >&2 echo "ERROR: no obs ID provided"
    exit 1
fi

#strip the last / if present in output directory filepath
outdir=${outdir%/}
echo Using output directory: $outdir

s3_path=${s3_path%/}
echo Using output S3 location: $s3_path

#create output directory with full permissions
if [ -d "$outdir" ]; then
    sudo chmod -R 777 $outdir
else
    sudo mkdir -m 777 $outdir
fi

#create uvfits download location with full permissions
if [ -d /uvfits ]; then
    sudo chmod -R 777 /uvfits
else
    sudo mkdir -m 777 /uvfits
fi

# Check if the uvfits file exists locally; if not, download it from S3
if [ ! -f "/uvfits/${obs_id}.uvfits" ]; then

    # Check that the uvfits file exists on S3
    uvfits_exists=$(aws s3 ls ${uvfits_s3_loc}/${obs_id}.uvfits)
    if [ -z "$uvfits_exists" ]; then
        >&2 echo "ERROR: uvfits file not found"
        echo "Job Failed"
        exit 1
    fi

    # Download uvfits from S3
    sudo aws s3 cp ${uvfits_s3_loc}/${obs_id}.uvfits \
    /uvfits/${obs_id}.uvfits --quiet

    # Verify that the uvfits downloaded correctly
    if [ ! -f "/uvfits/${obs_id}.uvfits" ]; then
        >&2 echo "ERROR: downloading uvfits from S3 failed"
        echo "Job Failed"
        exit 1
    fi
fi

# Run python catalog script
python ~/MWA/MJW-MWA/Long_Run/TV_Split.py ${obs_id} /uvfits/${obs_id}.uvfits $outdir $TV_min $TV_max $cal_min $cal_max $chan

# Move rfi outputs to S3
i=1  #initialize counter
aws s3 mv ${outdir}/ ${s3_path}/ --recursive \
--exclude "*" --include "*${obs_id}*" --quiet
while [ $? -ne 0 ] && [ $i -lt 10 ]; do
    let "i += 1"  #increment counter
    >&2 echo "Moving FHD outputs to S3 failed. Retrying (attempt $i)."
    aws s3 mv ${outdir}/ ${s3_path}/ \
    --recursive --exclude "*" --include "*${obs_id}*" --quiet
done

# Remove uvfits and metafits from the instance
sudo rm /uvfits/${obs_id}.uvfits

# Copy gridengine stdout to S3
aws s3 cp ~/grid_out/TV_Split_Job_AWS.sh.o${JOB_ID} \
${s3_path}/grid_out/TV_Split_Job_AWS.sh.o${JOB_ID}_${myip}.txt \
--quiet

# Copy gridengine stderr to S3
aws s3 cp ~/grid_out/TV_Split_Job_AWS.sh.e${JOB_ID} \
${s3_path}/grid_out/TV_Split_Job_AWS.sh.e${JOB_ID}_${myip}.txt \
--quiet
