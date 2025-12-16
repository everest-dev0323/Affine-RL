import os
import requests

ENVIRONMENT_NAMES = ['SAT', 'ABD-V2', 'DED-V2', 'ALFWORLD', 'WEBSHOP', 'BABYAI', 'SCIWORLD', 'TEXTCRAFT']

def get_sampling_range():
    """Get the sampling range from the environment"""
    response = requests.get(f"{os.getenv('AFFINE_ENDPOINT')}/config/environments")
    return response.json()['param_value']
