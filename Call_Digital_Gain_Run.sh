#!/bin/bash
#$ -V
#$ -N Broadband_Drill
#$ -S /bin/bash

echo JOBID ${JOB_ID}
echo TASKID ${SGE_TASK_ID}

python /nfs/eor-00/h1/mwilensk/MJW-MWA/Digital_Gain_Run.py ${SGE_TASK_ID}
