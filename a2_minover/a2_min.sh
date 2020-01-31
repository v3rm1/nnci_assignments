#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=a2_minover
#SBATCH --mem=1500
mv slurm-* sbatch_log/
module load Python/3.6.4-foss-2018a
python3 perceptron_minover.py
