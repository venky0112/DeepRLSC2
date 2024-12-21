#!/bin/bash
#SBATCH --account=project_2001281
#SBATCH --partition=small
#SBATCH --mem=64G
#SBATCH --time=0-8 
#SBATCH --mail-user=nicola.dainese@aalto.fi
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80

# email research-support@csc.fi for help
module load pytorch/nvidia-20.03-py3
singularity_wrapper exec python run.py $*

