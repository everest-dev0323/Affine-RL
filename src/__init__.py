def get_base_task_name(env_name: str) -> str:
    """Extract and normalize task name, removing version suffixes.
    
    Examples:
        "affine:sat" -> "sat"
        "affine:ded-v2" -> "ded"
        "affine:abd-v2" -> "abd"
    """
    # Extract env name from template (e.g., "affine:sat" -> "sat")
    env_name = env_name.split(":", 1)[1] if ":" in env_name else env_name
    # Remove version suffix if present (e.g., "ded-v2" -> "ded")
    if "-v" in env_name:
        env_name = env_name.rsplit("-v", 1)[0]
    return env_name