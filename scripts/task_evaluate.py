#!/usr/bin/env python3
"""
Generate datasets for print using trajectory generation.

"""

import os
import sys
import json
import random
import argparse
import asyncio
from tqdm import tqdm
import concurrent.futures
import threading


# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/environments/primeintellect/lgc-v2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/environments/trace'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../AffineFoundation/affinetes/affinetes/templates'))

from games.dyck_language.generator import DyckLanguageGenerator
from games.dyck_language2.generator import DyckLanguage2Generator
from games.game_of_24.generator import GameOf24Generator
from games.operation.generator import OperationGenerator
from games.cryptarithm.generator import CryptarithmGenerator
from games.boolean_expressions.generator import BooleanExpressionsGenerator
from games.sudoku.generator import SudokuGenerator

import trace_task

dyck_language_generator = DyckLanguageGenerator()
dyck_language2_generator = DyckLanguage2Generator()
game_of_24_generator = GameOf24Generator()
operation_generator = OperationGenerator()
cryptarithm_generator = CryptarithmGenerator()
boolean_expressions_generator = BooleanExpressionsGenerator()
sudoku_generator = SudokuGenerator()

generator_classes = {
    "dyck_language": dyck_language_generator,
    "dyck_language2": dyck_language2_generator,
    "game_of_24": game_of_24_generator,
    "operation": operation_generator,
    "cryptarithm": cryptarithm_generator,
    "boolean_expressions": boolean_expressions_generator,
    "sudoku": sudoku_generator,
}

DATASET_RANGE = 23303

def logic_evaluate(task_type: str, response: str, answer: str):
    generator = generator_classes.get(task_type)
    resp_answer = generator.extract_answer(response)
    ground_truth_answer = generator.extract_answer(answer)

    if resp_answer == ground_truth_answer:
        return 1.0
    else:
        return 0.0
    
def print_evaluat(response: str, answer: str):
    """
    Evaluate trace response
    """
    score = 0

    cleaned_prediction = trace_task.clean_llm_prediction(response)
    ground_truth = answer
    score += 1.0 if trace_task.compare_outputs(ground_truth, cleaned_prediction) else 0.0
    
    return score
