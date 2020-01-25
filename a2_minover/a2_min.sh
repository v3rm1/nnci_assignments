#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=a2_minover
#SBATCH --mem=4000
module load Python/3.6.4-foss-2018a
python perceptron_minover.py