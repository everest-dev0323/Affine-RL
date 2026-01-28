# Data Processing

### Affine Dataset Download & Upload
```bash
hf download --repo-type dataset hongyu05/affine-dataset --local-dir affine-dataset
hf upload --repo-type dataset hongyu05/affine-dataset affine-dataset
```

### Dataset Generation

#### Hex
``` bash
python scripts/data/generate_katahex_dataset.py \
    --output_file affine-dataset/openspiel/hex_golden.jsonl \
    --processed_log_file log/data/openspiel/hex_processed.log \
    --workers 8
    
python scripts/data/generate_katahex_dataset.py \
    --output_file affine-dataset/openspiel/hex_determine.jsonl \
    --processed_log_file log/data/openspiel/hex_processed_determine.log \
    --workers 100 \
    --random \
    --num_random 10000
```

#### Clobber
``` bash
python scripts/data/generate_clobber_golden_dataset.py \
    --output_file affine-dataset/openspiel/clobber_golden.jsonl \
    --processed_log_file log/data/openspiel/clobber_processed.log \
    --workers 8
    
python scripts/data/generate_clobber_golden_dataset.py \
    --output_file affine-dataset/openspiel/clobber_determine.jsonl \
    --processed_log_file log/data/openspiel/clobber_processed_determine.log \
    --workers 100 \
    --random \
    --num_random 10000
```





