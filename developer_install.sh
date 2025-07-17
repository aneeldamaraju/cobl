#!/bin/bash

module load python/3.10.12-fasrc01
ENV_NAME="CoBL"

# Check if the Conda environment exists
if conda env list | grep -qE "^$ENV_NAME\s"; then
    echo "Conda environment '$ENV_NAME' exists."
else
    echo "Creating new conda environment '$ENV_NAME' with Python 3.10..."
    conda create -y -n "$ENV_NAME" python=3.10
fi

# Activate the environment
source activate "$ENV_NAME"
./install.sh