```bash
pip install -U peft wandb deepspeed trl[vllm] 
pip install flash-attn --no-build-isolation
```

hf download --repo-type dataset hongyu05/RL_2_6 --local-dir dataset/grpo


```
ray start --head --num-gpus 8
```
