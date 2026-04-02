# Logistic Regression 实验报告

## 1. 问题定义

本实验实现的是二分类 Logistic Regression。对每个样本，输入特征为 $x_i \in \mathbb{R}^d$，标签为 $y_i \in \{0,1\}$。模型先计算线性打分，再通过 Sigmoid 函数映射为类别 1 的概率：

$$
z_i = w^T x_i + b
$$

$$
\hat{y}_i = P(y_i=1\mid x_i;w,b)=\sigma(z_i)=\frac{1}{1+e^{-z_i}}
$$

其中 $w\in\mathbb{R}^d$ 是权重，$b\in\mathbb{R}$ 是偏置项。

为了书写统一，后文常把偏置并入参数向量。定义增广特征和增广参数：

$$
\tilde{x}_i = \begin{bmatrix}x_i \\ 1\end{bmatrix},\quad
\tilde{w} = \begin{bmatrix}w \\ b\end{bmatrix}
$$

于是

$$
z_i = \tilde{w}^T \tilde{x}_i
$$

这种写法与代码中的做法一致：先在特征矩阵末尾拼接一列常数 1，再把参数写成一个向量。

## 2. 从单样本推导损失函数

对单个样本 $(x_i, y_i)$，Logistic Regression 假设标签服从伯努利分布：

$$
P(y_i\mid x_i;w,b)=\hat{y}_i^{y_i}(1-\hat{y}_i)^{1-y_i}
$$

取对数似然：

$$
\log P(y_i\mid x_i;w,b)=y_i\log \hat{y}_i+(1-y_i)\log(1-\hat{y}_i)
$$

训练时通常最小化负对数似然，也就是二元交叉熵损失：

$$
\ell_i(w,b)=-\Big[y_i\log \hat{y}_i+(1-y_i)\log(1-\hat{y}_i)\Big]
$$

把 $\hat{y}_i=\sigma(z_i)$ 代入后，单样本损失可写成

$$
\ell_i(w,b)
=-\Big[y_i\log \sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))\Big]
$$

这就是整个模型优化的基本单位。多样本情况下，只需要把每个样本的损失加总或求平均即可。

## 3. 从单样本到求和形式

对 $m$ 个样本，整体经验风险写为

$$
J(w,b)=\frac{1}{m}\sum_{i=1}^{m}\ell_i(w,b)
$$

展开后得到

$$
J(w,b)
=-\frac{1}{m}\sum_{i=1}^{m}
\Big[y_i\log \hat{y}_i+(1-y_i)\log(1-\hat{y}_i)\Big]
$$

这是最常见的标量求和形式。它的优点是直观，能够清楚地表达“每个样本都贡献一个损失，再取平均”的思想。

如果加入 $L_2$ 正则化，目标函数变为

$$
J_{reg}(w,b)=J(w,b)+\frac{\lambda}{2m}\lVert w\rVert_2^2
$$

注意一般不对偏置 $b$ 做正则化，因此正则项只作用在 $w$ 上。

## 4. 多样本多特征的矩阵形式

### 4.1 数据矩阵与参数向量

设共有 $m$ 个样本，每个样本 $d$ 维。把所有样本堆叠成矩阵：

$$
X=\begin{bmatrix}
x_1^T\\
x_2^T\\
\vdots\\
x_m^T
\end{bmatrix}\in\mathbb{R}^{m\times d},\quad
y=\begin{bmatrix}y_1\\y_2\\\vdots\\y_m\end{bmatrix}\in\mathbb{R}^{m}
$$

若使用增广特征，则令

$$
\tilde{X}=\begin{bmatrix}
\tilde{x}_1^T\\
\tilde{x}_2^T\\
\vdots\\
\tilde{x}_m^T
\end{bmatrix}\in\mathbb{R}^{m\times(d+1)},\quad
\tilde{w}\in\mathbb{R}^{d+1}
$$

为了简化记号，下面直接写成 $X$ 和 $w$，默认偏置已经并入参数。

### 4.2 从逐样本到矩阵计算

对每个样本都有

$$
z_i=w^T x_i
$$

把所有样本的线性打分一次性写出来，就是矩阵乘法：

$$
z=Xw
$$

其中 $z\in\mathbb{R}^m$，第 $i$ 个分量正好是 $z_i$。再逐元素施加 Sigmoid：

$$
\hat{y}=\sigma(z)=\sigma(Xw)
$$

这里的 $\sigma$ 作用于向量时表示按元素计算。

于是求和形式的损失可以写成紧凑的矩阵形式：

$$
J(w)=-\frac{1}{m}\Big[y^T\log \hat{y}+(1-y)^T\log(1-\hat{y})\Big]
$$

其中对数也按元素计算，再与向量做内积。展开后与上一节的求和形式完全等价：

$$
y^T\log \hat{y}=\sum_{i=1}^{m} y_i\log \hat{y}_i
$$

$$
(1-y)^T\log(1-\hat{y})=\sum_{i=1}^{m}(1-y_i)\log(1-\hat{y}_i)
$$

因此矩阵形式不是新的目标函数，而只是对标量求和形式的高效重写。

## 5. 直接用矩阵形式推导梯度

这一部分给出不依赖逐样本展开的推导。设

$$
z=Xw,\quad \hat{y}=\sigma(z)
$$

并记单个样本损失为

$$
\ell_i=-\Big[y_i\log \hat{y}_i+(1-y_i)\log(1-\hat{y}_i)\Big]
$$

先对单个样本求导。因为

$$
\frac{d\sigma(z_i)}{dz_i}=\sigma(z_i)(1-\sigma(z_i))=\hat{y}_i(1-\hat{y}_i)
$$

所以

$$
\frac{\partial \ell_i}{\partial z_i}
=\hat{y}_i-y_i
$$

这个结果是 Logistic Regression 里最关键的简化之一。它说明“交叉熵 + Sigmoid”组合后，关于线性打分 $z_i$ 的梯度非常简洁。

由于 $z_i=w^T x_i$，故

$$
\frac{\partial \ell_i}{\partial w}
=\frac{\partial \ell_i}{\partial z_i}\frac{\partial z_i}{\partial w}
=(\hat{y}_i-y_i)x_i
$$

对所有样本求和并取平均后：

$$
\nabla_w J(w)=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)x_i
$$

把每一项堆叠成矩阵，就得到向量化形式：

$$
\nabla_w J(w)=\frac{1}{m}X^T(\hat{y}-y)
$$

如果加入 $L_2$ 正则化，则梯度变为

$$
\nabla_w J_{reg}(w)=\frac{1}{m}X^T(\hat{y}-y)+\frac{\lambda}{m}w
$$

对应偏置项的梯度为

$$
\nabla_b J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)
$$

若把偏置并入增广参数，则这一个式子也可以统一写成对增广参数的矩阵梯度。

## 6. 这是凸优化

### 6.1 单样本损失的凸性

为了说明整体目标是凸的，先看单样本损失。把损失改写成只关于 $z_i$ 的形式：

$$
\ell_i(z_i)=-\Big[y_i\log \sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))\Big]
$$

利用 Sigmoid 的性质，可以化简为更便于分析的形式：

$$
\ell_i(z_i)=\log(1+e^{z_i})-y_i z_i
$$

对 $z_i$ 求一阶导数：

$$
\frac{d\ell_i}{dz_i}=\sigma(z_i)-y_i
$$

再求二阶导数：

$$
\frac{d^2\ell_i}{dz_i^2}=\sigma(z_i)(1-\sigma(z_i))
$$

由于对任意 $z_i$，都有 $0<\sigma(z_i)<1$，所以

$$
\sigma(z_i)(1-\sigma(z_i))\ge 0
$$

因此，**根据我们在人工智能数学基础中学习的二阶充要性条件**，单样本损失关于 $z_i$ 是凸函数。

### 6.2 对参数 $w$ 的凸性

因为 $z_i=w^Tx_i$ 是关于参数 $w$ 的仿射函数，而凸函数与仿射函数复合后仍保持凸性，所以 $\ell_i(w)$ 对 $w$ 也是凸的。

再看整体目标函数：

$$
J(w)=\frac{1}{m}\sum_{i=1}^{m}\ell_i(w)
$$

凸函数的非负加权和仍然是凸函数，因此 $J(w)$ 是凸函数。

如果加入 $L_2$ 正则化：

$$
J_{reg}(w)=J(w)+\frac{\lambda}{2m}\lVert w\rVert_2^2
$$

由于 $\lVert w\rVert_2^2$ 也是凸函数，所以正则化后的目标仍然凸。

### 6.3 用 Hessian 进一步说明

把所有样本一起看，记

$$
\hat{y}=\sigma(Xw)
$$

则 Hessian 可以写成

$$
\nabla^2_w J(w)=\frac{1}{m}X^T R X
$$

其中

$$
R=\operatorname{diag}\big(\hat{y}_1(1-\hat{y}_1),\dots,\hat{y}_m(1-\hat{y}_m)\big)
$$

因为 $R$ 是对角线上元素非负的对角矩阵，所以它是半正定的。任取任意向量 $v$，有

$$
v^T\nabla^2_w J(w)v
=\frac{1}{m}(Xv)^TR(Xv)\ge 0
$$

这说明 Hessian 半正定，因此 $J(w)$ 是凸优化问题。

如果再加上 $L_2$ 正则化，则 Hessian 变为

$$
\nabla^2_w J_{reg}(w)=\frac{1}{m}X^TRX+\frac{\lambda}{m}I
$$

在 $\lambda>0$ 时，它通常会让目标函数更强凸，优化过程也更稳定。

## 7. 优化算法与矩阵化更新

由于目标函数是凸的，使用梯度下降就能够稳定地逼近全局最优解。一次完整更新写成矩阵形式就是：

$$
w \leftarrow w - \eta \nabla_w J(w)
$$

代入梯度公式：

$$
w \leftarrow w - \eta \cdot \frac{1}{m}X^T(\hat{y}-y)
$$

若带正则化，则更新变为

$$
w \leftarrow w - \eta\left(\frac{1}{m}X^T(\hat{y}-y)+\frac{\lambda}{m}w\right)
$$

偏置项同理：

$$
b \leftarrow b - \eta \cdot \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)
$$

如果使用增广向量表示，则可以把 $w$ 和 $b$ 合并为一个参数向量，从而把更新完全统一到一条矩阵公式里。

