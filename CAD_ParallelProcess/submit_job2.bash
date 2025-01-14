#!/bin/bash
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=3
#SBATCH --time=15:59:59
#SBATCH --output=/home/%u/scratch/peymannr/Results/slurm-%u_%A_%a.out
#SBATCH --error=/home/%u/scratch/peymannr/Results/slurm-%u_%A_%a.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=p.nazarirobati@uleth.ca
#SBATCH --array=17

conda activate env
conda create -n env
conda activate env
conda install python==3.9.6
pip install numpy
pip install scipy
pip install pandas
pip install quantities 
pip install neo
pip install elephant
pip install multiprocess


echo "Starting task: $SLURM_ARRAY_TASK_ID"

cd ~/projects/def-tatsuno/peymannr/codes
python main_assembly_detection1.py $SLURM_ARRAY_TASK_ID

#echo "Input file:" $((SLURM_ARRAY_TASK_ID)).pkl
#input_file=$((SLURM_ARRAY_TASK_ID)).pkl
####file=$(ls *.pkl | sed -n "${SLURM_ARRAY_TASK_ID}p")

