# CodEOE

Data and code for paper "CODEOE: A BENCHMARK FOR JOINTLY EXTRACTING CROSS-DOCUMENT EVENTS AND OPINIONS FROM SOCIAL MEDIA".

## Overview
In this work, we propose a new task named CodEOE, which aims to jointly extract Events and Opinions from multiple documents. More details about the task can be found in our paper.

<p align="center">
<img src="./sample/example.png" width="40%">
</p>

## CodEOE Data
The dataset can be found at:
```bash
data/dataset
    - en
    - zh
```
## Code Usage

+ Train && Evaluate on the Chinese dataset
  ```bash
  bash scripts/train_zh.sh
  ```
+ Train && Evaluate on the English dataset
  ```bash
  bash scripts/train_en.sh
  ```

+ Hyperparameters:
You can set hyperparameters in `main.py` or `src/config.yaml`. `main.py` has a higher priority.

