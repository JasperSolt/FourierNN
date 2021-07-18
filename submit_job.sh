#!/bin/bash

#SBATCH --time=24:00:00

#SBATCH -p gpu-condo --gres=gpu:1
#SBATCH --account=jpober-condo
#SBATCH --constraint=p100

##SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=150G

#SBATCH -J Fourier_NN

#SBATCH -o Fourier_NN.out
#SBATCH -e Fourier_NN.out

# Set up the environment by loading modules
source torchenv/bin/activate

# Run a script
python -u train_Fourier_NN.py
python -u predict_Fourier_NN.py
python -u plot_model_results.py

