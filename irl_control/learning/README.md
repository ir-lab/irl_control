# Environment Variables
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export PATH=/opt/homebrew/opt/llvm/bin:$PATH
export CC=/opt/homebrew/opt/llvm/bin/clang

# Conda Environment
conda activate google

# Python Package
mujoco_py==2.1.2.14

# Install irl_control
pip install -e .