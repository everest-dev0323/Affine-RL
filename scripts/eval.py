import asyncio

from src.evaluation import evaluate_one, evaluate_env

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B" # "WebScraper991923/Affine-S5" # 

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

result = asyncio.run(
    evaluate_one(model, "game", task_id=400000021, tokenizer=tokenizer)
)

print(result)