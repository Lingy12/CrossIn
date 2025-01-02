# CrossIn: An Efficient Instruction Tuning Approach for Cross-Lingual Knowledge Alignment

## Authors

Geyu Lin, Bin Wang, Zhengyuan Liu, Nancy F. Chen

## Overview

This repository contains the code for the paper "CrossIn: An Efficient Instruction Tuning Approach for Cross-Lingual Knowledge Alignment".

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Build CrossIn data

```bash
bash sample_scripts/build_data.sh ## You need to download the actual data from alpaca and playtpus into data/ folder first
```

2. Run training

```bash
bash sample_scripts/run_training.sh <dataset_name> <stage> <exp_group> <prompt> <batch> <epoch> <lr>
```

3. Evaluation

The evaluation is done based on https://github.com/SeaEval/SeaEval/tree/SeaEval_v0.1.

## Citation

```
@misc{lin2024crossinefficientinstructiontuning,
      title={CrossIn: An Efficient Instruction Tuning Approach for Cross-Lingual Knowledge Alignment}, 
      author={Geyu Lin and Bin Wang and Zhengyuan Liu and Nancy F. Chen},
      year={2024},
      eprint={2404.11932},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.11932}, 
}
```