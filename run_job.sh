#!/bin/bash -l
#SBATCH --job-name=splitFair

# speficity number of nodes
#SBATCH -N 1
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1

#SBATCH --ntasks-per-node=1

# specify the walltime e.g 20 mins
#SBATCH -t 90:00:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joana.tirana@ucdconnect.ie

# run from current directory
cd $SLURM_SUBMIT_DIR
#=/home/people/21211297/scratch/SplitFair

# command to use
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/software/anaconda/3.2021.05/anaconda3/lib/
source ~/miniconda3/bin/activate
conda activate myenv
#python -m pip install torch torchvision torchaudio
#python -m pip install flwr-datasets==0.3.0
#pip install -U pillow
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/software/anaconda/3.2021.05/anaconda3/lib/
python test_whole.py
