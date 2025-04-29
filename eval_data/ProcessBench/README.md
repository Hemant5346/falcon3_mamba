---
language:
- en
tags:
- math
- reasoning
size_categories:
- 1K<n<10K
license: apache-2.0
configs:
- config_name: default
  data_files:
  - split: gsm8k
    path: "gsm8k.json"
  - split: math
    path: "math.json"
  - split: olympiadbench
    path: "olympiadbench.json"
  - split: omnimath
    path: "omnimath.json"
---
# ProcessBench

This repository contains the dataset of the [ProcessBench](https://huggingface.co/papers/2412.06559) benchmark proposed by Qwen Team.

You can refer to our [GitHub repository](https://github.com/QwenLM/ProcessBench) for the evaluation code and the prompt templates we use in this work.

If you find this work relevant or helpful to your work, please kindly cite us:

```
@article{processbench,
  title={ProcessBench: Identifying Process Errors in Mathematical Reasoning}, 
  author={
    Chujie Zheng and Zhenru Zhang and Beichen Zhang and Runji Lin and Keming Lu and
    Bowen Yu and Dayiheng Liu and Jingren Zhou and Junyang Lin
  },
  journal={arXiv preprint arXiv:2412.06559},
  year={2024}
}
```

## Data Usage

You can use the following code to preview the dataset:

```python
import json
from datasets import load_dataset

dataset = load_dataset('Qwen/ProcessBench', split='gsm8k')
print(json.dumps(dataset[0], indent=2))

# Expected output:
"""
{
  "id": "gsm8k-0",
  "generator": "Qwen2-7B-Instruct",
  "problem": "Sue lives in a fun neighborhood...",
  "steps": [
    "To find out how many more pink plastic flamingos were out than...",
    ...
  ],
  "final_answer_correct": false,
  "label": 1
}
"""
```
