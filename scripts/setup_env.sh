#!/bin/bash

# Docker images and their port mappings
declare -A IMAGES=(
    ["bignickeye/agentgym:webshop-v2"]="36001:8000"
    ["bignickeye/agentgym:alfworld-v2"]="36002:8000"
    ["bignickeye/agentgym:sciworld-v2"]="36003:8000"
    ["bignickeye/agentgym:textcraft-v2"]="36004:8000"
    ["bignickeye/agentgym:babyai-v2"]="36005:8000"
)

# Container name prefix
CONTAINER_PREFIX="agentgym"

echo "=== AgentGym Docker Environment Setup ==="

# Function to get container name from image
get_container_name() {
    local image=$1
    local env_name=$(echo "$image" | sed 's/.*:\(.*\)-v2/\1/')
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

