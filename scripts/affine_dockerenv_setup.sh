#!/bin/bash

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

# Docker images and their port mappings (using ports from .env)
declare -A IMAGES=(
    ["bignickeye/affine:v4"]="${AFFINE_PORT}:8000"
    ["bignickeye/affine:sci"]="${AFFINE_SCI_PORT}:8000"
    ["bignickeye/affine:mth"]="${AFFINE_MTH_PORT}:8000"
    ["bignickeye/affine:lgc"]="${AFFINE_LGC_PORT}:8000"
    ["bignickeye/affine:cde"]="${AFFINE_CDE_PORT}:8000"
)

# Container name prefix
CONTAINER_PREFIX="affine"

echo "=== Affine Docker Environment Setup ==="

# Function to get container name from image
get_container_name() {
    local image=$1
    local env_name=$(echo "$image" | sed 's/.*://')
    echo "${CONTAINER_PREFIX}-${env_name}"
}

# Pull images if they don't exist and run containers
for image in "${!IMAGES[@]}"; do
    port_mapping="${IMAGES[$image]}"
    container_name=$(get_container_name "$image")

    echo ""
    echo "--- Processing: $image ---"

    # Check if image exists locally
    if ! docker image inspect "$image" &> /dev/null; then
        echo "Image not found locally. Pulling $image..."
        docker pull "$image"
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to pull $image"
            continue
        fi
    else
        echo "Image $image already exists locally."
    fi

    # Get host port
    host_port=$(echo "$port_mapping" | cut -d':' -f1)

    # Check if port is already in use by any container
    if docker ps --format '{{.Ports}}' | grep -q "0.0.0.0:${host_port}->"; then
        echo "Port $host_port is already in use. Skipping."
        continue
    fi

    # Check if container exists but is stopped
    if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        echo "Container $container_name exists but is stopped. Starting it..."
        docker start "$container_name"
    else
        # Run new container
        echo "Starting new container $container_name on port $port_mapping..."
        docker run -d --name "$container_name" -p "$port_mapping" "$image"
    fi

    if [ $? -eq 0 ]; then
        echo "Container $container_name started successfully."
    else
        echo "ERROR: Failed to start container $container_name"
    fi
done

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Running containers:"
docker ps --filter "name=${CONTAINER_PREFIX}" --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}"
