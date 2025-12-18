import ast
import json

def process_mathador(example):
    info = json.loads(example["info"])
    game_data = json.loads(info["game_data_str"])
    solution = game_data["metadata"]["solutions"][0]
    s2 = "{" + solution.replace("nums", "'nums'").replace(", ops", ", 'ops'") + "}"
    # But easier: rewrite the string:
    parsed = ast.literal_eval(s2)
    length = len(parsed["ops"])
    answer = str(parsed["nums"][0])
    upper_level = ["*", '/']
    lower_level = ["+", '-']
    for i in range(length):
        if i > 0:
            if parsed["ops"][i] in upper_level and parsed["ops"][i - 1] in lower_level:
                answer = "(" + answer + ")" + parsed["ops"][i] + str(parsed["nums"][i + 1])
                continue
        answer = answer + parsed["ops"][i] + str(parsed["nums"][i + 1])
    # print(answer)
    answer = f"```python{answer}```"
    return {
        "question": example["question"],
        "answer": answer,
    }

def process_zebra_puzzle(example):
    info = json.loads(example["info"])
    game_data = json.loads(info["game_data_str"])
    answer = game_data["metadata"]["answer"]
    answer = f"```json\n{json.dumps(answer)}\n```"
    return {
        "question": example["question"],
        "answer": answer,
    }

def process_arc_agi(example):
    info = json.loads(example["info"])
    game_data = json.loads(info["game_data_str"])
    answer = game_data["answer"]
    answer = f"```json\n{json.dumps(answer)}```"
    return {
        "question": example["question"],
        "answer": answer,
    }

def process_web_of_lies(example):
    info = json.loads(example["info"])
    game_data = json.loads(info["game_data_str"])
    answer = game_data["answer"]
    answer = f"The answer is **{answer}**."
    return {
        "question": example["question"],
        "answer": answer,
    }

def process_math_path(example):
    info = json.loads(example["info"])
    game_data = json.loads(info["game_data_str"])
    answer = game_data["answer"]
    answer = f"[[{answer}\n]]"
    return {
        "question": example["question"],
        "answer": answer,
    }

def process_kukurasu(example):
    info = json.loads(example["info"])
    game_data = json.loads(info["game_data_str"])
    answer = game_data["metadata"]["solution"]
    answer = f"```python\n{json.dumps(answer)}\n```"
    return {
        "question": example["question"],
        "answer": answer,
    }

def process_buggy_tables(example):
    info = json.loads(example["info"])
    game_data = json.loads(info["game_data_str"])
    answer = game_data["answer"]
    answer = """```json{"result": [{"answer": """ + answer + """}]}```"""
    return {
        "question": example["question"],
        "answer": answer,
    }

def process_number_wall(example):
    info = json.loads(example["info"])
    game_data = json.loads(info["game_data_str"])
    answer = game_data["metadata"]["solution"]
    answer = f"```python\n{json.dumps(answer)}\n```"
    return {
        "question": example["question"],
        "answer": answer,
    }

def process_minesweeper(example):
    info = json.loads(example["info"])
    game_data = json.loads(info["game_data_str"])
    answer = game_data["metadata"]["current_mines"]
    return {
        "question": example["question"],
        "answer": str(answer),
    }


def send_python_meta_solution(example):
    info = json.loads(example["info"])
    game_data = json.loads(info["game_data_str"])
    answer = game_data["metadata"]["solution"]
    answer = f"```python\n{answer}\n```"
    return {
        "question": example["question"],
        "answer": answer,
    }
    
def send_python_raw_data(example):
    info = json.loads(example["info"])
    game_data = json.loads(info["game_data_str"])
    answer = game_data["answer"]
    answer = f"```python\n{answer}\n```"
    return {
        "question": example["question"],
        "answer": answer,
    }

def send_raw_answer(example):
    info = json.loads(example["info"])
    game_data = json.loads(info["game_data_str"])
    answer = game_data["answer"]
    return {
        "question": example["question"],
        "answer": str(answer),
    }

def send_meta_solution(example):
    info = json.loads(example["info"])
    game_data = json.loads(info["game_data_str"])
    answer = game_data["metadata"]["solution"]
    return {
        "question": example["question"],
        "answer": str(answer),
    }

def send_boxed_answer(example):
    info = json.loads(example["info"])
    game_data = json.loads(info["game_data_str"])
    answer = game_data["answer"]
    answer = f"\\boxed{{{answer}}}"
    return {
        "question": example["question"],
        "answer": answer,
    }

python_meta_solution_task = ["arrow_maze"]
python_raw_data_task = ["cipher", "goods_exchange", "sudoku", "survo"]
raw_answer_task = ["calcudoko", "dyck_language", "dyck_language_errors", "futoshiki", "time_sequence", "wordscapes"]
meta_solution_task = ["campsite", "numbrix", "norinori"]
boxed_answer = ["object_counting", "object_properties", "word_sorting", "word_sorting_mistake", "space_reasoning_tree", "boolean_expressions"]

def process_logic(example):
    info = json.loads(example["info"])
    task = info["task"]
    data = None
    if task in python_meta_solution_task:
        data = send_python_meta_solution(example)
    elif task in python_raw_data_task:
        data = send_python_raw_data(example)
    elif task in raw_answer_task:
        data = send_raw_answer(example)
    elif task in meta_solution_task:
        data = send_meta_solution(example)
    elif task in boxed_answer:
        data = send_boxed_answer(example)
    elif task == "mathador":
        data = process_mathador(example)
    elif task == "zebra_puzzle":
        data = process_zebra_puzzle(example)
    elif task == "arc_agi":
        data = process_arc_agi(example)
    elif task == "web_of_lies":
        data = process_web_of_lies(example)
    elif task == "math_path":
        data = process_math_path(example)
    elif task == "kukurasu":
        data = process_kukurasu(example)
    elif task == "buggy_tables":
        data = process_buggy_tables(example)
    elif task == "number_wall":
        data = process_number_wall(example)
    elif task == "minesweeper":
        data = process_minesweeper(example)
    else:
        print(example)
        data = example
    return data
