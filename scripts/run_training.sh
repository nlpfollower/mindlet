#!/bin/bash
# run_training.sh - Script to run TorchTitan training with tensor preloading
set -e

print_usage() {
    echo "Usage: $0 [--config <config_file>] [--preload <true|false>] [--script <script_file>]"
    echo "  --config <config_file>: Path to training configuration file"
    echo "  --preload <true|false>: Enable or disable tensor preloading (default: true)"
    echo "  --script <script_file>: Path to a script to run the TorchTitan training"
    exit 1
}

# Parse arguments
CONFIG_FILE=""
PRELOAD=true
SCRIPT_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --preload)
            PRELOAD="$2"
            shift 2
            ;;
        --script)
            SCRIPT_FILE="$2"
            shift 2
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

# Verify config file
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Config file must be specified with --config"
    print_usage
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Installing jq..."
    sudo yum install -y jq
fi

# Read configuration file
MODEL_PATH=$(jq -r '.model_path' "$CONFIG_FILE")
DATASET_PATH=$(jq -r '.dataset_path' "$CONFIG_FILE")
TENSOR_PRELOAD_ENABLED=$(jq -r '.tensor_preload.enabled' "$CONFIG_FILE")
TENSOR_PRELOAD_THREADS=$(jq -r '.tensor_preload.threads' "$CONFIG_FILE")
REDIS_HOST=$(jq -r '.tensor_preload.redis_host' "$CONFIG_FILE")
REDIS_PORT=$(jq -r '.tensor_preload.redis_port' "$CONFIG_FILE")
RUN_ID=$(jq -r '.tensor_preload.run_id' "$CONFIG_FILE")
TORCHTITAN_CONFIG=$(jq -r '.torchtitan_config.config_path // ""' "$CONFIG_FILE")
RANK=$(jq -r '.rank' "$CONFIG_FILE")
WORLD_SIZE=$(jq -r '.world_size' "$CONFIG_FILE")
HEAD_NODE=$(jq -r '.node_topology.head' "$CONFIG_FILE")
OUTPUT_DIR=$(jq -r '.output_dir' "$CONFIG_FILE")
NODE_IP=$(jq -r '.node_ip' "$CONFIG_FILE")

# Validate required fields
if [ -z "$MODEL_PATH" ] || [ "$MODEL_PATH" == "null" ]; then
    echo "Error: model_path is required in the config file"
    exit 1
fi

if [ -z "$DATASET_PATH" ] || [ "$DATASET_PATH" == "null" ]; then
    echo "Error: dataset_path is required in the config file"
    exit 1
fi

if [ -z "$RANK" ] || [ "$RANK" == "null" ]; then
    echo "Error: rank is required in the config file"
    exit 1
fi

if [ -z "$WORLD_SIZE" ] || [ "$WORLD_SIZE" == "null" ]; then
    echo "Error: world_size is required in the config file"
    exit 1
fi

if [ -z "$HEAD_NODE" ] || [ "$HEAD_NODE" == "null" ]; then
    echo "Error: node_topology.head is required in the config file"
    exit 1
fi

if [ -z "$NODE_IP" ] || [ "$NODE_IP" == "null" ]; then
    echo "Error: node_ip is required in the config file"
    exit 1
fi

if [ -z "$RUN_ID" ] || [ "$RUN_ID" == "null" ]; then
    echo "Error: run ID is required in the config file"
    exit 1
fi

echo "Node IP: $NODE_IP (Rank $RANK of $WORLD_SIZE)"
echo "Model path: $MODEL_PATH"
echo "Dataset path: $DATASET_PATH"
echo "Head node: $HEAD_NODE"

# Set up clean exit trap
trap cleanup EXIT INT TERM
cleanup() {
    echo "Cleaning up processes..."
    if [ -f /tmp/training.pid ]; then
        pid=$(cat /tmp/training.pid)
        if kill -0 $pid 2>/dev/null; then
            echo "Killing training process $pid"
            kill $pid
        fi
    fi

    if [ -f /tmp/tensor_preloader.pid ]; then
        pid=$(cat /tmp/tensor_preloader.pid)
        if kill -0 $pid 2>/dev/null; then
            echo "Killing tensor preloader process $pid"
            kill $pid
        fi
    fi

    rm -f /tmp/training.pid /tmp/tensor_preloader.pid
    echo "Cleanup complete"
}

# Handle Redis for head node
if [ "$RANK" == "0" ]; then
    echo "This is the head node. Ensuring Redis is running..."

    # Check if Redis is already running
    if ! docker ps | grep -q redis; then
        echo "Starting Redis container..."
        docker run --name redis -p $REDIS_PORT:6379 -d redis

        # Give Redis a moment to initialize
        sleep 2
    else
        echo "Redis is already running"
        # Flush Redis to clear any previous data
        docker exec redis redis-cli FLUSHALL
    fi

    # Verify Redis is accepting connections
    if ! docker exec redis redis-cli ping | grep -q PONG; then
        echo "Error: Redis is not responding properly"
        exit 1
    fi

    echo "Redis is ready on port $REDIS_PORT"
else
    # For worker nodes, verify Redis is available on the head node
    echo "Worker node: Verifying Redis is available on $REDIS_HOST:$REDIS_PORT..."

    # Check if Redis is reachable
    max_attempts=30
    attempt=1
    redis_ready=false

    while [ $attempt -le $max_attempts ]; do
        if nc -z -w 1 $REDIS_HOST $REDIS_PORT; then
            echo "Successfully connected to Redis on $REDIS_HOST:$REDIS_PORT"
            redis_ready=true
            break
        fi

        echo "Waiting for Redis to be available... (attempt $attempt/$max_attempts)"
        sleep 1
        attempt=$((attempt + 1))
    done

    if [ "$redis_ready" = false ]; then
        echo "Error: Could not connect to Redis after $max_attempts attempts"
        exit 1
    fi
fi

# Start tensor preloader if enabled
if [ "$TENSOR_PRELOAD_ENABLED" == "true" ] && [ "$PRELOAD" == "true" ]; then
    echo "Using tensor preloader managed by mindlet with run ID: $RUN_ID"
else
    echo "Tensor preloading is disabled"
fi

# Set up environment for TorchTitan
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_IB_DISABLE=1

# If a script file is provided, execute it
if [ -n "$SCRIPT_FILE" ]; then
    if [ ! -f "$SCRIPT_FILE" ]; then
        echo "Error: Script file '$SCRIPT_FILE' not found"
        exit 1
    fi

    echo "Executing provided script file: $SCRIPT_FILE"
    chmod +x "$SCRIPT_FILE"

    # Run the script and capture its PID
    "$SCRIPT_FILE" &
    TRAIN_PID=$!
    echo $TRAIN_PID > /tmp/training.pid
    echo "Training started with PID $TRAIN_PID"

    # Wait for the training to complete
    wait $TRAIN_PID
    EXIT_CODE=$?

    echo "Training exited with status $EXIT_CODE"
    exit $EXIT_CODE
fi

# If no script file is provided, run the classic way
echo "Starting training..."
cd "$HOME/workspace/torchtitan"

# Construct and run the torchrun command
torchrun \
    --nproc_per_node=8 \
    --nnodes="$WORLD_SIZE" \
    --node_rank="$RANK" \
    --master_addr="$HEAD_NODE" \
    --master_port=29500 \
    --rdzv_id=101 \
    --rdzv_backend=c10d \
    train.py \
    --training.dataset_path="$DATASET_PATH" \
    --model.tokenizer_path="$MODEL_PATH/tokenizer.model" \
    --job.dump_folder="$OUTPUT_DIR" \
    --checkpoint.use_tensor_preload \
    --checkpoint.preload_run_id="$RUN_ID" &

TRAIN_PID=$!
echo $TRAIN_PID > /tmp/training.pid
echo "Training started with PID $TRAIN_PID"

# Wait for the training to complete
wait $TRAIN_PID
EXIT_CODE=$?

echo "Training exited with status $EXIT_CODE"
exit $EXIT_CODE