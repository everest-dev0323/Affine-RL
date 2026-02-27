```bash
pip install -U peft wandb deepspeed trl[vllm] 
pip install flash-attn --no-build-isolation
```

```bash
ray start --head --num-gpus 8
```
