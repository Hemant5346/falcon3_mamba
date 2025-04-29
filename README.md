# SCAN-PRM

## Dataset

- SCAN-Base in `datasets/SCAN-Base`
- SCAN-Pro in `datasets/SCAN-Pro`

Due to the storage limit of a single file, we upload the first 100 rows.

## Step 1: Env build

```
conda create -n scan-prm python=3.10
conda activate scan-prm
pip install torch
pip install flash-attn --no-build-isolation --no-cache-dir
pip install vllm
pip install -r requirements.txt
```

## Step 2: PRM Data Synthesis

Note that you can jump to the Step 3 to directly train the PRM using the provided datasets in `dataset/`.

```bash
# Data Synthesis
python -m src.eval_prm.main_datasyn

# Convert to standard dataset
cd SCAN-Base && python process.py
```

Note that you should manually set some parameters in `src/eval_prm/main_datasyn.py`


## Step 3: Train PRMs

```bash
bash scripts/train.sh
```

## Step 4: Eval PRMs

```bash
# Best-Of-N Evaluation
python -m src.eval_prm.main_bon

# ProcessBench Evaluation
python -m src.eval_prm.main_pb
```

Note that you should set the path of the trained process reward models in the scripts.

## Others

We also provide the synthesis scripts of our preliminary study.

```bash
python -m src.eval_prm.main_rollout_eval
```

