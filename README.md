<div align="center">

# 🌍 CrossIn

### An Efficient Instruction Tuning Approach for Cross-Lingual Knowledge Alignment

[![arXiv](https://img.shields.io/badge/arXiv-2404.11932-b31b1b.svg)](https://arxiv.org/abs/2404.11932)

</div>

## 📋 Overview

CrossIn is a novel approach for efficient instruction tuning that focuses on cross-lingual knowledge alignment. This repository contains the official implementation of our paper "CrossIn: An Efficient Instruction Tuning Approach for Cross-Lingual Knowledge Alignment".

## 👥 Authors

- Geyu Lin
- Bin Wang
- Zhengyuan Liu
- Nancy F. Chen

## 🚀 Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage Guide

#### 1️⃣ Build CrossIn Data
```bash
bash sample_scripts/build_data.sh
```
> **Note**: You need to download the Alpaca and Platypus datasets into the `data/` folder first.

#### 2️⃣ Training
```bash
bash sample_scripts/run_training.sh <dataset_name> <stage> <exp_group> <prompt> <batch> <epoch> <lr>
```

#### 3️⃣ Evaluation
Evaluation is performed using the [SeaEval framework](https://github.com/SeaEval/SeaEval/tree/SeaEval_v0.1).

## 📚 Citation

If you find this work useful, please consider citing our paper:

```bibtex
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

