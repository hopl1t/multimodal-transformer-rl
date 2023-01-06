# Multi-Modal-Transformer-RL


[<img src="https://img.shields.io/badge/license-MIT-blue">]
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)



MMTRL combines both audio and video modalities in RL tasks over decision transformers to achive SOTA performence in two multimodal settings: Minecraft2d and Skeleton+ (stereo version)

It is a step forward combining these two papers: [Multimodal Reinforcement Learning with Effective State Representation Learning](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1684.pdf) and [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) together with better audio processing.

The online training phase of this repo was originally forked from [CleanRL](https://github.com/vwxyzjn/cleanrl)
The offline uses the DecisionTransformer architecture as per [DecisionTransformer](https://github.com/kzl/decision-transformer)

## Get started

Prerequisites:
* Python >=3.7.1,<3.10 (not yet 3.10)
* [Poetry 1.2.1+](https://python-poetry.org)

To run experiments locally, give the following a try:

```bash
```

To use experiment tracking with wandb, run
```bash
```
