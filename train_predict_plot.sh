#!/bin/bash

source torchenv/bin/activate
python3 train_Fourier_NN.py
python3 predict_Fourier_NN.py
python3 plot_model_results.py