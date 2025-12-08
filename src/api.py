import os
import requests
from dotenv import load_dotenv

from config import ENVIRONMENT_NAMES

load_dotenv()

def get_task_range():
    """Get the task range from the environment"""
    
    task_ranges = {}
    AFFINE_ENDPOINT = os.getenv("AFFINE_ENDPOINT")
    
    for env_name in ENVIRONMENT_NAMES:
        response = requests.get(f"{AFFINE_ENDPOINT}/pool/uid/0/{env_name}")
        task_ranges[env_name] = response.json()['sampling_range']
        
    return task_ranges
    