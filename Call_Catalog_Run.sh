#! /bin/bash
#$ -V
#$ -N P2_RFI
#$ -S /bin/bash

echo JOBID ${JOB_ID}
echo TASKID ${SGE_TASK_ID}

python /nfs/eor-00/h1/mwilensk/MJW-MWA/Catalog_Run.py ${SGE_TASK_ID}
