```bash
conda create -n kt-sft python=3.11
conda activate kt-sft
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
# pip install -r requirements/metrics.txt --no-build-isolation
# pip install "transformers==4.55.0"

pip install cpufeature
pip install ktransformers --no-build-isolation
pip install flash-attn --no-build-isolation
```


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

Merge LoRA Weights
```bash
llamafactory-cli export --model_name_or_path openai/gpt-oss-20b \
    --adapter_name_or_path saves/gpt-20b/lora/sft \
    --export_dir gpt_merged
```