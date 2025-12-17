#!/bin/bash

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../../.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

# AgentGym Docker images and their port mappings
declare -A AGENTGYM_IMAGES=(
    ["affinefoundation/agentgym:textcraft"]="${AGENTGYM_TEXTCRAFT_PORT}:8000"
    ["affinefoundation/agentgym:sciworld"]="${AGENTGYM_SCIWORLD_PORT}:8000"
    ["affinefoundation/agentgym:babyai"]="${AGENTGYM_BABYAI_PORT}:8000"
    ["affinefoundation/agentgym:webshop"]="${AGENTGYM_WEBSHOP_PORT}:8000"
    ["affinefoundation/agentgym:alfworld"]="${AGENTGYM_ALFWORLD_PORT}:8000"
)

# Affine Docker images and their port mappings
declare -A AFFINE_IMAGES=(
    ["affinefoundation/affine-env:v4"]="${AFFINE_PORT}:8000"
    ["affinefoundation/sci:pi"]="${AFFINE_SCI_PORT}:8000"
    ["affinefoundation/mth:pi"]="${AFFINE_MTH_PORT}:8000"
    ["affinefoundation/lgc:pi"]="${AFFINE_LGC_PORT}:8000"
    ["affinefoundation/cde:pi"]="${AFFINE_CDE_PORT}:8000"
    ["affinefoundation/swebench:pro"]="${AFFINE_SWE_PORT}:8000"
)

echo "=== Docker Environment Setup ==="

# Function to get container name from image
get_container_name() {
    local image=$1
    local prefix=$2
    # Extract the image name part (after last / and before :)
    # e.g., affinefoundation/sci:pi -> sci
    # e.g., affinefoundation/agentgym:textcraft -> textcraft (uses tag for agentgym)
    local repo_name=$(echo "$image" | sed 's|.*/||' | sed 's/:.*//') # e.g., sci, mth, agentgym
    local tag=$(echo "$image" | sed 's/.*://')                       # e.g., pi, textcraft

    # For agentgym images, use the tag as the env name
    # For other images (sci, mth, lgc, cde), use the repo name
    if [[ "$repo_name" == "agentgym" || "$repo_name" == "affine-env" ]]; then
        echo "${prefix}-${tag}"
    else
        echo "${prefix}-${repo_name}"
    fi
}

# Function to process images
process_images() {
    local -n images=$1
    local prefix=$2

    for image in "${!images[@]}"; do
        port_mapping="${images[$image]}"
        container_name=$(get_container_name "$image" "$prefix")

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
}

echo ""
echo "=== Setting up AgentGym Environments ==="
process_images AGENTGYM_IMAGES "agentgym"

echo ""
echo "=== Setting up Affine Environments ==="
process_images AFFINE_IMAGES "affine"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Running AgentGym containers:"
docker ps --filter "name=agentgym" --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}"
echo ""
echo "Running Affine containers:"
docker ps --filter "name=affine" --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}"
