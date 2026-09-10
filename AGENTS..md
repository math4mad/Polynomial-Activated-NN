# Polynomial-Activated NN for Fashion-MNIST

我们在 **Fashion-MNIST** 这个小数据集上，用 PyTorch 实现一个**多项式激活函数神经网络**，并与传统 MLP 做对比。下面这个 `AGENT.md` 文件可以作为你的项目实施指南，放在项目根目录下即可。

***

# 项目名称：Polynomial-Activated NN for Fashion-MNIST

## 项目概述

本项目旨在探索**可学习多项式基函数作为神经网络激活函数**的可行性，并在 Fashion-MNIST 数据集上验证其**参数效率**和**收敛速度**是否优于传统 ReLU 激活的 MLP。

## 技术栈

* Python 3.10+
* PyTorch 2.0+（启用 MPS 后端，适配 M1 Pro MacBook 10 core cpu, 16 core gpu, 32GB+512GB）
* torchvision（加载 Fashion-MNIST 数据集）
* Matplotlib（可视化训练曲线和多项式基函数）
* scipy&#x20;
* polars for dataframe operation
* great\_tables for  pretty table&#x20;

## 核心架构

### 对比模型

| **模型**        | **激活函数**                                        | **描述**       |
| ------------- | ----------------------------------------------- | ------------ |
| **MLP\_ReLU** | ReLU                                            | 标准全连接网络，作为基线 |
| **MLP\_Poly** | 可学习的 Jacobi/Hermite / Chebyshev / Bernstein 多项式 | 目标模型，研究参数效率  |

### 多项式激活网络设计

* **输入层**：784 个神经元（28×28 图像展平）
* **单隐藏层**：神经元数量可调（建议从 128 开始，对比 ReLU 模型的 256 神经元）
* **输出层**：10 个类别（Softmax）

### 多项式激活函数实现

```python
import torch
import torch.nn as nn

class HermiteActivation(nn.Module):
    """可学习的 Hermite 多项式激活函数，阶数可调"""
    def __init__(self, degree=4):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(degree + 1) * 0.1)
        self.degree = degree
        
    def forward(self, x):
        # Hermite 多项式递归：H_0=1, H_1=2x, H_{n+1}=2xH_n - 2nH_{n-1}
        H = [torch.ones_like(x), 2 * x]
        for n in range(1, self.degree):
            H.append(2 * x * H[-1] - 2 * n * H[-2])
        # 截断到指定阶数
        H = H[:self.degree + 1]
        # 加权求和
        result = sum(c * h for c, h in zip(self.coeffs, H))
        return result
```

## 构建与运行命令

| **任务** | **命令**                                                       | **说明**             |
| ------ | ------------------------------------------------------------ | ------------------ |
| 创建环境   | local python env                                             | 使用 Python 3.14.7   |
| 安装依赖   | `pip install torch torchvision matplotlib`                   | 注意 PyTorch 需支持 MPS |
| 运行训练   | `python train.py --activation poly --hidden 128 --epochs 20` | 多项式激活网络            |
| 运行基线   | `python train.py --activation relu --hidden 256 --epochs 20` | ReLU 基线网络          |
| 对比实验   | `python compare.py --poly_hidden 128 --relu_hidden 256`      | 自动化对比              |
| 可视化    | `python visualize.py`                                        | 绘制训练曲线和基函数图        |

## 代码规范

* **文件命名**：`snake_case.py`，如 `train.py`、`models.py`
* **函数命名**：`snake_case`，如 `train_epoch()`、`compute_accuracy()`
* **类命名**：`PascalCase`，如 `HermiteActivation`、`PolyMLP`
* **注释**：函数和类使用 docstring，关键逻辑使用行内注释
* **类型提示**：所有函数参数和返回值标注类型

## 实验流程

### 步骤 1：数据准备

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)
```

### 步骤 2：模型定义

实现两个模型：

* `MLP_ReLU(hidden_size)`：标准两层 MLP，ReLU 激活
* `PolyMLP(hidden_size, poly_degree=4)`：使用多项式激活函数，神经元数量可减少

### 步骤 3：训练循环

* 优化器：Adam（学习率 0.001）
* 损失函数：交叉熵损失
* 批次大小：64
* 训练轮数：20

### 步骤 4：对比分析

* **参数数量**：统计两个模型的总参数量
* **测试精度**：在测试集上评估
* **收敛速度**：绘制训练损失曲线
* **可视化多项式基函数**：输出多项式激活函数在 \[-3, 3] 区间上的形状

## 预期结果

* 多项式激活网络在 **参数减少 50%-70%** 的情况下，达到与 ReLU 网络相当的测试精度（约 88%-90%）
* 多项式网络的训练损失**下降更快**，收敛更早
* 可学习的多项式系数会形成**光滑、非线性**的激活函数形状

## 安全性

* 不提交任何敏感数据或 API 密钥
* 所有实验在本地 M1 Pro 上离线运行，无数据泄露风险
* 确保 `data/` 目录加入 `.gitignore`


##  publist as quarto note style ref  ./agents/quarto-publisth/SKILL.MD
## 参考资料

* DeepBern-Nets：用可学习 Bernstein 多项式激活函数实现参数缩减 70%-99.9%
* Orthogonal Polynomials in Neural Networks：Hermite、Chebyshev 等正交多项式作为激活函数的理论分析
* PyTorch MPS 文档：[https://pytorch.org/docs/stable/notes/mps.html](https://pytorch.org/docs/stable/notes/mps.html)

***

​

​

​
