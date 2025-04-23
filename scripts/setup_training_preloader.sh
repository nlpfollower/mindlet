#!/bin/bash
# setup_tensor_preloader.sh
# Script to set up tensor preloader environment for TorchTitan training
set -e

print_usage() {
    echo "Usage: $0 [--install-deps] [--build-only]"
    echo "  --install-deps: Install system dependencies (default: false)"
    echo "  --build-only: Only build the tensor preloader extension (default: false)"
    exit 1
}

# Parse arguments
INSTALL_DEPS=0
BUILD_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-deps)
            INSTALL_DEPS=1
            shift
            ;;
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        --help)
            print_usage
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Install system dependencies if requested
if [ $INSTALL_DEPS -eq 1 ]; then
    echo "Installing system dependencies..."
    sudo yum update -y
    sudo yum install -y gcc-c++ cmake git hiredis hiredis-devel mdadm

    # Install Docker for Redis if not already installed
    if ! command -v docker &> /dev/null; then
        echo "Installing Docker..."
        sudo amazon-linux-extras install docker -y
        sudo systemctl enable docker
        sudo systemctl start docker
        sudo usermod -aG docker $USER
        # Need to log out and log back in for group changes to take effect
        # For script purposes, we'll continue without requiring a logout
    fi
fi

WORKSPACE_DIR="$HOME/workspace"
mkdir -p $WORKSPACE_DIR

# Clone repositories if not in build-only mode
if [ $BUILD_ONLY -ne 1 ]; then
    echo "Setting up Gloo library..."
    # Make sure SSH key is loaded
    if ! ssh-add -l | grep -q "nlpfollower"; then
        eval $(ssh-agent -s)
        ssh-add ~/nlpfollower
    fi

    # Clone TorchTitan if not already present
    if [ ! -d "$WORKSPACE_DIR/torchtitan" ]; then
        echo "Cloning TorchTitan repository..."
        cd $WORKSPACE_DIR
        git clone git@github.com:nlpfollower/torchtitan.git
    fi

    # Clone Gloo if not already present
    if [ ! -d "$HOME/gloo" ]; then
        echo "Cloning Gloo repository..."
        cd $HOME
        git clone https://github.com/facebookincubator/gloo.git
    fi

    # Build Gloo
    echo "Building Gloo library..."
    cd $HOME/gloo
    mkdir -p build && cd build
    cmake -DBUILD_TEST=OFF -DUSE_REDIS=ON -DBUILD_TEST=OFF -DBUILD_EXAMPLES=OFF ..
    make -j$(nproc)
fi

# Build TorchTitan tensor preloader extension
echo "Building tensor preloader extension..."
cd $WORKSPACE_DIR/torchtitan/torchtitan/csrc
export GLOO_INCLUDE_DIR=$HOME/gloo
export GLOO_LIB_DIR=$HOME/gloo/build/gloo
export CFLAGS="-I/usr/local/cuda-12.6/include"
export CXXFLAGS="-I/usr/local/cuda-12.6/include"
pip install -e .

echo "Setting CUDA environment variables..."
if ! grep -q "CUDA" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
fi

# Setup Redis container if not in build-only mode
if [ $BUILD_ONLY -ne 1 ]; then
    # Check if Redis is already running
    if ! docker ps | grep -q "redis"; then
        echo "Starting Redis container..."
        docker run --name redis -p 6379:6379 -d redis
    else
        echo "Redis is already running."
    fi
fi

echo "Tensor preloader setup complete!"
echo "You can now use model_loader.py to preload tensors."