# Data Processing

### Affine Dataset Download & Upload
```bash
hf download --repo-type dataset hongyu05/affine-dataset --local-dir affine-dataset
hf upload --repo-type dataset hongyu05/affine-dataset affine-dataset
```

### Dataset Generation

#### Hex

Katahex building
```bash
cd external/katahex/
mkdir build
cd build
sudo apt-get install libzip-dev
cmake -DUSE_BACKEND=CUDA -DMAX_BOARD_LEN=11 ../cpp 
make -j4
cd ..
pip install gdown
gdown 1xMvP_75xgo0271nQbmlAJ40rvpKiFTgP
```

```
# 1. Set the GPU to Exclusive Process mode
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

# 2. Start the MPS Daemon
nvidia-cuda-mps-control -d
```

``` bash
python scripts/data/generate_katahex_dataset.py \
    --output_file affine-dataset/openspiel/hex_golden.jsonl \
    --processed_log_file log/data/openspiel/hex_processed.log \
    --workers 8
    
python scripts/data/generate_katahex_dataset.py \
    --output_file affine-dataset/openspiel/hex_determine.jsonl \
    --processed_log_file log/data/openspiel/hex_processed_determine.log \
    --workers 50 \
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





