---
name: create-python-env
description: >-
  Create Local python evn 
license: Apache-2.0
compatibility: "Requires Python 3.14+"
metadata:
  author: math4mad
  version: "1.0.0"
  category: general purpose
  tags: ["python",".venv, local env"]
---

# quarto 

## Overview

create local env 

## Instructions

-  python3 -m venv .venv
-  activate evn   source .venv/bin/activate 

## install package 
* Python 3.10+
* PyTorch 2.0+（启用 MPS 后端，适配 M1 Pro MacBook 10 core cpu, 16 core gpu, 32GB+512GB）
* torchvision（加载 Fashion-MNIST 数据集）
* Matplotlib（可视化训练曲线和多项式基函数）
* scipy&#x20;
* polars for dataframe operation
* great_tables for  pretty table&#x20;

## pip 国内镜像（PyPI 官方源超时/不可达时）

```bash
pip install <pkg> -i https://pypi.tuna.tsinghua.edu.cn/simple
# 或写入全局配置：
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 备选：https://mirrors.aliyun.com/pypi/simple/
```

macOS 上 torch 需 arm64 wheel（`macosx_*_arm64`），Python 3.14 起请确认 wheel 存在：
`pip download --no-deps --dest /tmp/t torch` 先行验证。大包建议后台安装（nohup），避免超时中断。
  


## Request
 if  working good you can create  requirements.txt  file
 