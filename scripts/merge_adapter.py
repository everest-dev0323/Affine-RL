import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("Affineme/Affine-5GWqrs7hgWRjj3yMF9uM4fnzowsbYCGugofQ7Bu1AAJ3L9Qz", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Affineme/Affine-5GWqrs7hgWRjj3yMF9uM4fnzowsbYCGugofQ7Bu1AAJ3L9Qz")

# model = PeftModel.from_pretrained(model, "./outputs/model/train_epoch_1")
model = PeftModel.from_pretrained(model, "./models/qwen3-sgrpo-output/checkpoint-1600")

model = model.merge_and_unload()  # this returns merged model

model.save_pretrained("./models/merged_model", safe_serialization=True)
tokenizer.save_pretrained("./models/merged_model")
