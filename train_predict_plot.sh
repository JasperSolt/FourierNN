#!/bin/bash

#SBATCH -t=24:00:00
#SBATCH -m=150G

# Specify a job name:
#SBATCH -J Fourier_NN

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.out

source torchenv/bin/activate
python3 train_Fourier_NN.py
python3 predict_Fourier_NN.py
python3 plot_model_results.py