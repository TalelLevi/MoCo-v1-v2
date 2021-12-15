#!/bin/bash

# Setup env
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate deep
echo "start clf job"
nvidia-smi
python main_clf.py
echo "done job"