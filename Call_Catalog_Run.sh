#! /bin/bash
#$ -V
#$ -N Temp_Run
#$ -S /bin/bash

echo JOBID ${JOB_ID}
echo TASKID ${SGE_TASK_ID}

python /nfs/eor-00/h1/mwilensk/MJW-MWA/Temp_Study.py ${SGE_TASK_ID}
