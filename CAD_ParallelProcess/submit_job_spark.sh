#!/bin/bash
#SBATCH --mem=64G               
#SBATCH --cpus-per-task=8       
#SBATCH --time=15:59:59
#SBATCH --output=/home/%u/scratch/peymannr/Results/spark-%u_%A_%a.out
#SBATCH --error=/home/%u/scratch/peymannr/Results/spark-%u_%A_%a.error
#SBATCH --array=1-20              # Array for batch tasks

module load java
module load python
module load spark

conda activate env

echo "Starting task: $SLURM_ARRAY_TASK_ID"

cd ~/projects/def-tatsuno/peymannr/codes

spark-submit \
  --master local[8] \
  --driver-memory 16G \
  main_assembly_detection_spark.py
