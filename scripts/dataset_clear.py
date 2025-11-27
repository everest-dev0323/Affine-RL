import json
import time

def clear_data(data):
    max_round = 1
    if "max_round" in data["evaluation"]["extra"]:
        max_round = data["evaluation"]["extra"]["max_round"]
    elif "max_round" in data:
        max_round = data["max_round"]
    
    if "prompt" in data["challenge"]:
        return {
            # "model": data["miner"]["model"],
            "env": data["challenge"]["env"],
            "task_id": data["challenge"]["challenge_id"],
            "score": data["evaluation"]["score"],
            "conversations": [{"role": "user", "content": data["challenge"]["prompt"]}, {"role": "assistant", "content": data["response"]["response"]}],
            "timestamp": data["challenge"]["timestamp"],
            "max_round": max_round
        }
    if "experiences" not in data["evaluation"]["extra"]["details"]:
        return
    if not data["challenge"]["env"].endswith("alfworld"):
        return
    if data["challenge"]["extra"]["task_id"] != 0:
        return
    
    return {
        # "model": data["miner"]["model"],
        "env": data["challenge"]["env"],
        "task_id": data["challenge"]["extra"]["task_id"],
        "score": data["evaluation"]["score"],
        "conversations": data["evaluation"]["extra"]["details"]["experiences"][0]["conversation"] if str(type(data["evaluation"]["extra"]["details"]["experiences"])) == "<class 'list'>" else [{"role": "user", "content": data["evaluation"]["extra"]["details"]["experiences"]["challenge"]}, {"role": "assistant", "content": data["evaluation"]["extra"]["details"]["experiences"]["llm_response"]}],
        "timestamp": data["challenge"]["timestamp"],
        "max_round": max_round
    }

import requests

url = "https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev/affine/index.json"

response = requests.get(url)

index_list = json.loads(response.content.decode('utf-8'))
reversed_list = []
for index in index_list:
    reversed_list.append(index)

for index in reversed_list:
    print(index)
    url = "https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev/" + index

    try:
        response = requests.get(url, timeout=10)
    except Exception as e:
        print(e)
        time.sleep(60)
        response = requests.get(url, timeout=10)
        
    data = json.loads(response.content.decode('utf-8'))
    for item in data:
        if clear_data(item):
            with open("datasets/total.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(clear_data(item)) + "\n")

