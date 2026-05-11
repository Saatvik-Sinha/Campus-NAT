#!/usr/bin/env bash
# =============================================================================
# PointNAT - Fixed install script for CUDA 12.2 systems
# USAGE: source install_fixed.sh   (NOT bash install_fixed.sh)
# =============================================================================
#
# Changes from original install.sh:
#  - Removed 'module' HPC commands (not available on standard Linux)
#  - Replaced PyTorch 1.10.1+cu113 with PyTorch 2.1.2+cu121
#    (CUDA 12.2 is backward-compatible with cu121 wheels)
#  - Upgraded Python 3.7 → 3.9 (3.7 is EOL and conflicts with modern packages)
#  - Upgraded numpy=1.20 → numpy=1.24 (compatible with Python 3.9 + PyTorch 2.x)
#  - Updated torch-scatter URL to match PyTorch 2.1.2+cu121
#  - Fixed conda activate (works when sourced, not run as bash subshell)
#  - Updated scikit-learn to 1.3.x (1.0.2 fails on Python 3.9+)
# =============================================================================

# set -e to exit on first error

export TORCH_CUDA_ARCH_LIST="8.6"  # Change to match your GPU:
                                    # RTX 2080 Ti / Titan XP: 7.5
                                    # V100: 7.0
                                    # A100: 8.0
                                    # RTX 3090/3080: 8.6
                                    # RTX 4090: 8.9

# ---------------------------------------------------------------------------
# 0. Initialize conda in this shell (required before activate/deactivate)
# ---------------------------------------------------------------------------
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 1. Remove any old environment and create a fresh one
# ---------------------------------------------------------------------------
conda deactivate 2>/dev/null || true
conda env remove --name openpoints -y 2>/dev/null || true

conda create -n openpoints -y python=3.9 numpy=1.24 numba

conda activate openpoints

# ---------------------------------------------------------------------------
# 2. Install PyTorch 2.1.2 with CUDA 12.1 wheels
#    (These run correctly on CUDA 12.2 — NVIDIA guarantees backward compat)
# ---------------------------------------------------------------------------
conda install -y pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 \
    pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify torch + CUDA are working before continuing
python -c "import torch; print('PyTorch:', torch.__version__); \
           print('CUDA available:', torch.cuda.is_available()); \
           print('CUDA version:', torch.version.cuda)"

# ---------------------------------------------------------------------------
# 3. Install torch-scatter (updated URL for PyTorch 2.1.2 + cu121)
# ---------------------------------------------------------------------------
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# ---------------------------------------------------------------------------
# 4. Install pip requirements
#    Note: scikit-learn 1.0.2 is incompatible with Python 3.9+; 
#    override it to a working version.
# ---------------------------------------------------------------------------

# If requirements.txt pins scikit-learn==1.0.2, override it:
pip install scikit-learn==1.3.2

# Install the rest of requirements.txt (scikit-learn will already be satisfied)
pip install -r requirements.txt --ignore-installed scikit-learn 2>/dev/null || \
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 5. Build C++ / CUDA extensions
# ---------------------------------------------------------------------------

# 5a. PointNet++ batch ops
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

# 5b. Grid subsampling (needed for S3DIS sphere experiments)
cd subsampling
python setup.py build_ext --inplace
cd ..

# 5c. Point Transformer ops
cd pointops/
python setup.py install
cd ..

# 5d. Chamfer distance (optional, for completion tasks)
cd chamfer_dist
python setup.py install --user
cd ../

# 5e. Earth Mover Distance (optional, for completion tasks)
cd emd
python setup.py install --user
cd ../../../

echo ""
echo "============================================="
echo " PointNAT environment setup complete!"
echo " Activate with: conda activate openpoints"
echo "============================================="
