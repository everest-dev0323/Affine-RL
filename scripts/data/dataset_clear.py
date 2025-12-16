import json
import re
import requests

# 8 environments
ENVS = ["abd", "ded", "sat", "alfworld", "sciworld", "babyai", "textcraft", "webshop"]
# Affine envs (prompt/response format, convert to single round conversation)
AFFINE_ENVS = ["abd", "ded", "sat"]


def detect_env(raw_str):
    """Detect environment from raw string"""
    raw_lower = raw_str.lower()
    for env in ENVS:
        if f'"{env}"' in raw_lower or f":{env}" in raw_lower or f'"{env}:' in raw_lower:
            return env
    return None


def extract_field_str(raw_str, field_name):
    """Extract a string field value using regex"""
    # Match "field_name": "value" or "field_name":"value"
    pattern = rf'"{field_name}"\s*:\s*"([^"]*)"'
    match = re.search(pattern, raw_str)
    if match:
        return match.group(1)
    return None


def extract_field_number(raw_str, field_name):
    """Extract a numeric field value using regex"""
    # Match "field_name": number or "field_name":number
    pattern = rf'"{field_name}"\s*:\s*([0-9.]+)'
    match = re.search(pattern, raw_str)
    if match:
        try:
            val = match.group(1)
            return float(val) if '.' in val else int(val)
        except:
            return None
    return None


def extract_conversations_from_json(data):
    """Try to extract conversations from parsed JSON data"""
    conversations = None

    # Try different paths to find conversations
    # Path 1: evaluation.extra.details[0].experiences[0].conversation (new structure)
    try:
        if "evaluation" in data and "extra" in data["evaluation"]:
            extra = data["evaluation"]["extra"]
            if "details" in extra:
                details = extra["details"]
                # details is an array
                if isinstance(details, list) and len(details) > 0:
                    detail_item = details[0]
                    if "experiences" in detail_item:
                        exp = detail_item["experiences"]
                        if isinstance(exp, list) and len(exp) > 0:
                            if "conversation" in exp[0]:
                                conversations = exp[0]["conversation"]
                # details is a dict with experiences directly
                elif isinstance(details, dict) and "experiences" in details:
                    exp = details["experiences"]
                    if isinstance(exp, list) and len(exp) > 0:
                        if "conversation" in exp[0]:
                            conversations = exp[0]["conversation"]
                    elif isinstance(exp, dict):
                        # Single experience with challenge/llm_response format
                        if "challenge" in exp and "llm_response" in exp:
                            conversations = [
                                {"role": "user", "content": exp["challenge"]},
                                {"role": "assistant", "content": exp["llm_response"]}
                            ]
    except:
        pass

    # Path 2: extra.conversation
    if not conversations:
        try:
            if "extra" in data and "conversation" in data["extra"]:
                conversations = data["extra"]["conversation"]
        except:
            pass

    # Path 3: Affine env format - challenge.prompt + response.response
    if not conversations:
        try:
            if "challenge" in data and "prompt" in data["challenge"]:
                prompt = data["challenge"]["prompt"]
                response = data.get("response", {}).get("response", "")
                if prompt and len(prompt) > 10:
                    conversations = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
        except:
            pass

    return conversations


def extract_score(raw_str, data):
    """Extract score from raw string or parsed data"""
    # Try from parsed data first
    try:
        if "evaluation" in data and "score" in data["evaluation"]:
            return data["evaluation"]["score"]
    except:
        pass

    try:
        if "score" in data:
            return data["score"]
    except:
        pass

    # Fallback to string extraction
    return extract_field_number(raw_str, "score")


def extract_task_id(raw_str, data):
    """Extract task_id from raw string or parsed data"""
    # Try from parsed data - evaluation.extra.task_id (new structure)
    try:
        if "evaluation" in data and "extra" in data["evaluation"]:
            if "task_id" in data["evaluation"]["extra"]:
                return data["evaluation"]["extra"]["task_id"]
    except:
        pass

    # Try challenge.extra.task_id
    try:
        if "challenge" in data and "extra" in data["challenge"]:
            if "task_id" in data["challenge"]["extra"]:
                return data["challenge"]["extra"]["task_id"]
    except:
        pass

    # Try challenge.challenge_id
    try:
        if "challenge" in data and "challenge_id" in data["challenge"]:
            return data["challenge"]["challenge_id"]
    except:
        pass

    # Try top-level task_id
    try:
        if "task_id" in data:
            return data["task_id"]
    except:
        pass

    # Fallback to string extraction
    task_id = extract_field_str(raw_str, "task_id")
    if not task_id:
        task_id = extract_field_str(raw_str, "challenge_id")
    return task_id


def extract_max_round(raw_str, data):
    """Extract max_round from raw string or parsed data"""
    try:
        if "evaluation" in data and "extra" in data["evaluation"]:
            if "max_round" in data["evaluation"]["extra"]:
                return data["evaluation"]["extra"]["max_round"]
    except:
        pass

    try:
        if "max_round" in data:
            return data["max_round"]
    except:
        pass

    return extract_field_number(raw_str, "max_round")


def extract_reward(raw_str, data):
    """Extract reward from raw string or parsed data"""
    # Try evaluation.extra.details[0].reward (new structure)
    try:
        if "evaluation" in data and "extra" in data["evaluation"]:
            extra = data["evaluation"]["extra"]
            if "details" in extra:
                details = extra["details"]
                if isinstance(details, list) and len(details) > 0:
                    if "reward" in details[0]:
                        return details[0]["reward"]
                    # Also try details[0].experiences[0].reward
                    if "experiences" in details[0]:
                        exp = details[0]["experiences"]
                        if isinstance(exp, list) and len(exp) > 0 and "reward" in exp[0]:
                            return exp[0]["reward"]
    except:
        pass

    # Try evaluation.reward
    try:
        if "evaluation" in data and "reward" in data["evaluation"]:
            return data["evaluation"]["reward"]
    except:
        pass

    # Try top-level reward
    try:
        if "reward" in data:
            return data["reward"]
    except:
        pass

    return extract_field_number(raw_str, "reward")


def clear_data(raw_str, data):
    """Parse rollout data using string operations and JSON fallback"""

    # Detect environment
    env = detect_env(raw_str)
    if not env:
        return None

    # Extract required fields
    score = extract_score(raw_str, data)
    task_id = extract_task_id(raw_str, data)
    conversations = extract_conversations_from_json(data)

    # Skip if missing essential data
    if score is None or not conversations:
        return None

    # Build result
    result = {
        "env": env,
        "task_id": task_id,
        "score": score,
        "conversations": conversations
    }

    # Optional fields
    reward = extract_reward(raw_str, data)
    if reward is not None:
        result["reward"] = reward

    max_round = extract_max_round(raw_str, data)
    if max_round is not None:
        result["max_round"] = max_round

    # Try to get model name
    try:
        if "miner" in data and "model" in data["miner"]:
            result["model"] = data["miner"]["model"]
    except:
        pass

    # Try to get timestamp
    try:
        if "challenge" in data and "timestamp" in data["challenge"]:
            result["timestamp"] = data["challenge"]["timestamp"]
        elif "timestamp" in data:
            result["timestamp"] = data["timestamp"]
    except:
        pass

    return result


def main():
    import os

    url = "https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev/affine/index.json"
    output_dir = "./datasets/rollout"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("Fetching index...")
    response = requests.get(url)
    index_list = json.loads(response.content.decode('utf-8'))

    # Reverse the list to process from latest to oldest
    index_list = list(reversed(index_list))
    print(f"Total files to process: {len(index_list)} (reversed order - latest first)")

    success_count = 0
    fail_count = 0
    env_counts = {env: 0 for env in ENVS}

    for index in index_list:
        print(f"Processing: {index}")
        file_url = "https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev/" + index

        try:
            response = requests.get(file_url, timeout=10)
        except Exception as e:
            print(f"Error: {e}")
            input("Please reactivate vpn")
            print("Retrying...")
            response = requests.get(file_url, timeout=10)

        raw_content = response.content.decode('utf-8')
        data_list = json.loads(raw_content)

        # Generate output filename from index path
        # e.g., "affine/2024-01-01/data.json" -> "affine_2024-01-01_data.jsonl"
        output_filename = index.replace("/", "_").replace(".json", ".jsonl")
        output_path = os.path.join(output_dir, output_filename)

        file_success = 0
        file_fail = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for item in data_list:
                # Convert item back to string for string-based parsing
                item_str = json.dumps(item)
                result = clear_data(item_str, item)

                if result:
                    f.write(json.dumps(result) + "\n")
                    success_count += 1
                    file_success += 1
                    env_counts[result["env"]] += 1
                else:
                    fail_count += 1
                    file_fail += 1

        print(f"  -> Saved to {output_filename} (success: {file_success}, failed: {file_fail})")

    print(f"\n=== Summary ===")
    print(f"Total Success: {success_count}")
    print(f"Total Failed: {fail_count}")
    print(f"By environment:")
    for env, count in env_counts.items():
        print(f"  {env}: {count}")


if __name__ == "__main__":
    main()

