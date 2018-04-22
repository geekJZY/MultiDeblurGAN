#!/bin/bash
##NMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=continue_train_170     #Set the job name to "JobExample4"
#SBATCH --time=60:50:00                #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=5                  #Request 1 task
#SBATCH --mem=20000M                  #Request 2560MB (2.5GB) per node
#SBATCH --output=fullModelUnsupervised.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:2                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu




module load Anaconda/3-5.0.0.1 
source activate ML
python train.py --dataroot /scratch/user/jiangziyu/train/ --learn_residual --resize_or_crop crop --fineSize 256 --continue_train --epoch_count 220 --batchSize 16 --save_epoch_freq 2
echo job done!
