
```bash
# Example conversion (check KTransformers docs for exact command)
python -m ktransformers.convert \
    --input_path /path/to/kimi-k2.5-int4 \
    --output_path /path/to/kimi-k2.5-bf16 \
    --output_dtype bf16
```


```bash
# 1. Configure your dataset in the YAML
# Edit: scripts/sft/kimik25_kt_lora.yaml
# Change: dataset: your-dataset-name

# 2. Run training
USE_KT=1 llamafactory-cli train scripts/sft/kimik25_kt_lora.yaml

# 3. Verify with chat
llamafactory-cli chat scripts/sft/kimik25_kt_inference.yaml
```