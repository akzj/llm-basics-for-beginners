---
title: "首页"
date: 2025-01-01T00:00:00+08:00
draft: false
---

# LLM Basics for Beginners

## Transformer 核心公式

### Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### Multi-Head Attention

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)W^O
$$

其中：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

### 前馈神经网络

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

### Layer Normalization

$$
\text{LayerNorm}(x) = \alpha \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

## 数学补完计划

### 矩阵基础知识

矩阵是LLM中最基本的数据结构之一，理解矩阵运算对于掌握LLM的工作原理至关重要。

#### 矩阵定义

一个 \( m \times n \) 的矩阵 \( A \) 可以表示为：

$$
A = \begin{bmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    a_{21} & a_{22} & \cdots & a_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1} & a_{m2} & \cdots & a_{mn} \\
\end{bmatrix}
$$

#### 矩阵运算

1. **矩阵加法**：相同维度的矩阵对应元素相加
   $$ A + B = [a_{ij} + b_{ij}] $$

2. **矩阵乘法**：设 \( A \) 为 \( m \times p \) 矩阵，\( B \) 为 \( p \times n \) 矩阵，则
   $$ (AB)_{ij} = \sum_{k=1}^p a_{ik}b_{kj} $$

3. **转置**：将矩阵的行列互换
   $$ (A^T)_{ij} = a_{ji} $$

4. **逆矩阵**：若存在矩阵 \( B \) 使得 \( AB = BA = I \)，则 \( B = A^{-1} \)

#### 特殊矩阵

- **单位矩阵** \( I \)：对角线元素为1，其余为0
- **零矩阵** \( O \)：所有元素都为0
- **对称矩阵**：满足 \( A = A^T \)
- **对角矩阵**：仅对角线元素非零
- **正交矩阵**：满足 \( A^T A = AA^T = I \)

### Softmax 函数

Softmax函数是LLM中用于将向量转换为概率分布的关键函数：

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
$$

特性：
1. 输出值在(0, 1)之间
2. 所有输出值之和为1
3. 保持输入向量的相对大小关系

### 线性代数中的其他重要概念

1. **特征值与特征向量**：
   $$ Ax = \lambda x $$
   其中 \( \lambda \) 是特征值，\( x \) 是特征向量。

2. **奇异值分解 (SVD)**：
   $$ A = U\Sigma V^T $$
   其中 \( U \) 和 \( V \) 是正交矩阵，\( \Sigma \) 是对角矩阵。

3. **梯度下降**：
   $$ \theta = \theta - \alpha \nabla J(\theta) $$
   其中 \( \alpha \) 是学习率，\( \nabla J(\theta) \) 是损失函数的梯度。

## 相关资源

- [矩阵基础知识]({{< ref "/matrix_basics" >}})
- [Transformer 论文](https://arxiv.org/abs/1706.03762)
- [注意力机制详解](https://lilianweng.github.io/posts/2018-06-24-attention/)

## 联系方式

如有问题或建议，欢迎联系我们。