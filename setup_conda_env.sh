#!/bin/bash

# Verify conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Check NVIDIA drivers
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA drivers not found. Please install NVIDIA drivers first"
    exit 1
fi

# Environment name
ENV_NAME="tf-gpu"
PYTHON_VERSION="3.11"

echo "Setting up TensorFlow GPU environment..."

# Deactivate any active conda environments
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate

# Create and activate new environment
echo "Creating conda environment: $ENV_NAME"
conda create --name $ENV_NAME python=$PYTHON_VERSION -y || exit 1
conda activate $ENV_NAME

# Install TensorFlow with CUDA support
echo "Installing TensorFlow with CUDA support..."
pip install tensorflow[and-cuda]

# Setup environment variables
echo "Configuring environment variables..."
CONDA_ENV_PATH="$CONDA_PREFIX/etc/conda"
mkdir -p "$CONDA_ENV_PATH/activate.d"
mkdir -p "$CONDA_ENV_PATH/deactivate.d"

# Create activation script
ACTIVATION_SCRIPT="$CONDA_ENV_PATH/activate.d/env_vars.sh"
cat > "$ACTIVATION_SCRIPT" << 'EOF'
#!/bin/bash
export CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)"))
export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib:$LD_LIBRARY_PATH
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
EOF

# Create deactivation script
DEACTIVATION_SCRIPT="$CONDA_ENV_PATH/deactivate.d/env_vars.sh"
cat > "$DEACTIVATION_SCRIPT" << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
unset OLD_LD_LIBRARY_PATH
unset CUDNN_PATH
unset TF_ENABLE_ONEDNN_OPTS
unset TF_CPP_MIN_LOG_LEVEL
EOF

# Make scripts executable
chmod +x "$ACTIVATION_SCRIPT" "$DEACTIVATION_SCRIPT"

# Reload environment to apply changes
conda deactivate
conda activate $ENV_NAME

# Verify installation
echo "Verifying TensorFlow installation..."
python -c "
import tensorflow as tf
print('\n' + '-' * 20 + '\n')
print('TensorFlow version:', tf.__version__)
print('GPU devices:', tf.config.list_physical_devices('GPU'))
"

echo -e "\nSetup complete! To use this environment:"
echo "  - Activate: conda activate $ENV_NAME"
echo "  - Deactivate: conda deactivate"

# Install additional scientific packages
echo "Installing additional scientific packages..."
pip install scipy matplotlib numpy scikit-image pysteps opencv-contrib-python-headless

# TODO: Check if the implicit newline works or not
echo "Verifying additional packages..."
python -c "
import scipy
import matplotlib as mpl
import numpy as np
import skimage
import pysteps
import cv2
import pysteps
print('\nAdditional packages:')
print(f'SciPy version: {scipy.__version__}')
print(f'Matplotlib version: {mpl.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'SciKit-Image version: {skimage.__version__}')
print(f'PySTEPS version: {pysteps.__version__}')
print(f'OpenCV version: {cv2.__version__}')
"
