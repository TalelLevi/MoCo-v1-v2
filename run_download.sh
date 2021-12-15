#!/bin/bash

# Setup env
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate deep
echo "start download_data"
python download_data.py
echo "done download_data"