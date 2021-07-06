#!/bin/bash

#SBATCH --time=8:00:00

#SBATCH -p gpu-condo --gres=gpu:1
#SBATCH --account=jpober-condo
#SBATCH --constraint=p100
#SBATCH --mem=150G

#SBATCH -J Fourier_NN_test

#SBATCH -o Fourier_NN_test.out
#SBATCH -e Fourier_NN_test.out


# Set up the environment by loading modules
source torchenv/bin/activate

# Run a script
python -u train_Fourier_NN.py
python -u predict_Fourier_NN.py
python -u plot_model_results.py

