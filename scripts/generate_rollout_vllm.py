import sys
from vllm import LLM, SamplingParams
import random
import json
from tqdm import tqdm
import asyncio

sys.path.insert(0, 'AffineFoundation/affinetes/environments/primeintellect/lgc-v2')
sys.path.insert(0, 'AffineFoundation/affinetes/affinetes/templates')

from logic_task_v2 import LogicTaskV2

def parse_completion(output):
    # Extract the generated text from the output
    completion = output.outputs[0].text
    if "<think>" in completion:
        if "</think>" not in completion:
            return None
        else:
            start = completion.index("<think>") + len("<think>")
            end = completion.index("</think>")
            reasoning_content = completion[start:end].strip()
            content = completion[end + len("</think>"):].strip()
            return {"reasoning_content": reasoning_content, "content": content}
    else:
        if "</think>" in completion:
            reasoning_content = completion.split("</think>")[0].strip()
            content = completion.split("</think>")[-1].strip()
            return {"reasoning_content": reasoning_content, "content": content}
        else:
            return {"reasoning_content": "", "content": completion}

def generate_rollout_batch(batch_conversations, llm, sampling_params):
    outputs = llm.chat(batch_conversations, sampling_params=sampling_params)
    parsed_outputs = [parse_completion(output) for output in outputs]
    return parsed_outputs

def generate_dataset_batch(task, task_id_list, llm, sampling_params, batch_size):
    dataset = []
    for task_id in task_id_list:
        challenge = asyncio.run(task.generate(task_id))
        prompt = challenge.prompt
        conversation = [{"role": "user", "content": prompt}]
        dataset.append({"task_id": task_id, "conversation": conversation, "challenge": challenge})

    def create_mini_batches(data, batch_size):
        # Splits data into a list of lists
        return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    mini_batches = create_mini_batches(dataset, batch_size)

    for batch in tqdm(mini_batches, desc="Training Epoch 1"):
        # print(batch[0]["task_id"])
        # print(batch[0]["conversation"])
        # print(dataset)
        batch_conversations = [item["conversation"] for item in batch]
        outputs = generate_rollout_batch(batch_conversations, llm, sampling_params)
        print("Generated batch outputs.")
        for i, output in enumerate(outputs):
            item = batch[i]
            if output:
                score = asyncio.run(task.evaluate(output["content"], item["challenge"]))
                item["conversation"] = item["conversation"] + [{"role": "assistant", "reasoning_content": output["reasoning_content"], "content": output["content"]}]
                item["score"] = score
                new_item = {"task_id": item["task_id"], "conversation": item["conversation"], "score": item["score"]}
                with open("logic_reasoning_sota_rollout.jsonl", "a") as f:
                    f.write(json.dumps(new_item) + "\n")
            else:
                print(f"Skipping task_id {item['task_id']} due to incomplete generation.")

def main():
    # 1. Initialize the model (it will detect the chat template from the config)
    # model_name = "./Affine-H/merged_model_final"
    model_name = "eugene141759/affine-s-5CciarLspRcwjAunZGQEMP9VqDs82fPgZRuKLGFQhQVVfcda"
    llm = LLM(model=model_name)

    # # 2. Set sampling params
    sampling_params = SamplingParams(temperature=0.7, max_tokens=36768)

    task = LogicTaskV2()

    generation_count = 10
    task_id_list = []
    task_id_list += random.choices(range(0, 100_000_000), k=generation_count)
    task_id_list += random.choices(range(100_000_000, 200_000_000), k=generation_count)
    task_id_list += random.choices(range(200_000_000, 300_000_000), k=generation_count)
    task_id_list += random.choices(range(300_000_000, 400_000_000), k=generation_count)
    task_id_list += random.choices(range(400_000_000, 500_000_000), k=generation_count)
    task_id_list += random.choices(range(500_000_000, 600_000_000), k=generation_count)
    task_id_list += random.choices(range(600_000_000, 700_000_000), k=generation_count)
    task_id_list += random.choices(range(700_000_000, 800_000_000), k=generation_count)
    task_id_list += random.choices(range(800_000_000, 900_000_000), k=generation_count)
    # task_id_list += random.choices(range(0, 1_00_000_000), k=100)

    # print(task_id_list)

    # task_id_list = [80761638, 14780214, 25621711, 20539125, 49179915, 31872742, 28974007, 42234324, 33552936, 4680811, 81726075, 99225782, 97742697, 37696035, 85751783, 66717152, 16779714, 38379593, 32102260, 87693905, 29343967, 67696148, 63008630, 90500290, 25476100, 6928507, 4516050, 77722853, 69499302, 65462288, 10735500, 28534795, 21051373, 67673529, 43676004, 85629192, 73574972, 61031501, 8979612, 32576277, 56016258, 67466651, 78952536, 60623731, 91973623, 85354644, 12564511, 40834319, 1940948, 66247132, 70167959, 85257351, 8790410, 49009347, 38753865, 78984165, 60078792, 75002974, 35462715, 89979735, 92524362, 67890452, 23897559, 49687134, 85316480, 18169212, 35487960, 25397943, 64721998, 32803395, 42546713, 61887707, 37679007, 45755771, 51714610, 78988667, 81451641, 44127106, 71030457, 57695774, 36072470, 72498606, 47403471, 22417536, 9199436, 50214130, 10685961, 63790569, 6704615, 6667289, 33397547, 91371820, 20432920, 84603880, 33636992, 35470684, 5722174, 45876816, 60185401, 93512270]
    
    batch_size = 100
    generate_dataset_batch(task, task_id_list, llm, sampling_params, batch_size)

if __name__ == "__main__":
    main()
