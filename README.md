# ML-4

用 numpy 从零实现 Logistic Regression

## 1. 我理解的项目目标

这个 ML-4 的核心不是“调用现成库得到结果”，而是你要真正掌握二分类线性模型从数学到代码的完整闭环：

1. 明确模型假设：线性打分 + Sigmoid 概率映射。
2. 明确优化目标：最小化二元交叉熵（负对数似然）。
3. 用 numpy 手写训练流程：前向计算、损失、梯度、参数更新。
4. 用实验验证实现正确性：损失下降、指标提升、参数可解释。

如果最后只给出“能跑”的代码而没有训练曲线、实验对比、错误分析，通常不算完整完成。

## 2. 建议你按这个顺序做（高成功率）

## 2.1 先把数学写清楚

对每个样本 $x_i \in \mathbb{R}^d$，标签 $y_i \in \{0,1\}$：

$$
z_i = w^T x_i + b
$$

$$
\hat{y}_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}
$$

二元交叉熵损失：

$$
\mathcal{L}(w,b) = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]
$$

加上 L2 正则（可选但推荐）：

$$
\mathcal{L}_{reg}=\mathcal{L}+\frac{\lambda}{2N}\|w\|_2^2
$$

梯度（向量化）：

$$
\nabla_w = \frac{1}{N}X^T(\hat{y}-y) + \frac{\lambda}{N}w
$$

$$
\nabla_b = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)
$$

参数更新（梯度下降）：

$$
w \leftarrow w - \eta \nabla_w, \quad b \leftarrow b - \eta \nabla_b
$$

你报告里把上面这套写明，基本就把“理论正确性”交代清楚了。

## 2.2 再做工程实现（只用 numpy）

建议模块拆分：

1. 数据处理
- 读取数据
- 划分 train/val/test
- 特征标准化（只用 train 的统计量）

2. 模型核心
- `sigmoid(z)`：注意数值稳定（对 z 过大/过小时做裁剪）
- `forward(X)`：输出概率
- `compute_loss(y_hat, y)`：交叉熵，记得加 `eps`
- `compute_gradients(X, y_hat, y)`：向量化梯度
- `fit(...)`：训练循环
- `predict_proba(X)` / `predict(X, threshold=0.5)`

3. 实验与可视化
- 每轮记录 train loss、val loss、val accuracy
- 绘制损失曲线，观察是否收敛
- 输出最终 test 指标

## 2.3 最小可用版本（MVP）

优先先做一个“正确、稳定、可复现”的版本：

1. 固定随机种子。
2. 全部使用向量化，不写样本级 for 循环。
3. 先在小数据上验证损失单调下降。
4. 再上完整数据和调参。

## 2.4 进阶加分点（有余力再做）

1. 正则化对比：无正则 vs L2。
2. 学习率对比：$\eta$ 取不同量级。
3. 阈值分析：0.3/0.5/0.7 对 Precision-Recall 的影响。
4. 类别不平衡处理：class weight 或重采样。
5. 与 `sklearn.linear_model.LogisticRegression` 做结果对照（只用于验证，不用于主实现）。

## 3. 实验报告建议结构（你可以直接照着写）

1. 任务定义
- 数据集简介
- 二分类目标与评价指标

2. 方法
- Logistic Regression 数学推导
- 损失函数、梯度、优化算法
- 数值稳定性处理

3. 实验设置
- 数据划分方式
- 超参数（学习率、epoch、batch size、正则系数）
- 评价指标（Accuracy / Precision / Recall / F1 / AUC）

4. 实验结果
- 训练与验证曲线
- 测试集指标表格
- 对比实验与现象解释

5. 误差分析
- 错分样本类型
- 可能的数据问题或特征问题

6. 结论与改进
- 当前方案优点与局限
- 下一步可尝试方向

## 4. 容易踩坑的地方

1. 把标签写成 `{-1,1}` 却用 `0/1` 的 BCE 公式。
2. 没有标准化导致训练震荡或收敛慢。
3. `log(0)` 导致 `nan`（要加 `eps`）。
4. Sigmoid 溢出（`exp(1000)`）导致数值错误。
5. 数据泄漏：用全量数据统计均值方差。
6. 只看 Accuracy，不看 Precision/Recall/F1（尤其类别不平衡时）。

## 5. 参考资料（按优先级）

## 5.1 必读（先看这些就够完成项目）

1. CS229 课程笔记：Logistic Regression
- Andrew Ng 体系下的经典讲义，推导清晰，和课程作业风格很接近。

2. Stanford CS231n 的线性分类与损失函数部分
- 帮你把“概率解释、损失、优化”串起来。

3. 《Pattern Recognition and Machine Learning》（Bishop）第 4 章（线性模型用于分类）
- 理论更加严谨，适合报告写“方法部分”。

## 5.2 工程实现参考（用于代码正确性自查）

1. scikit-learn 文档：`LogisticRegression`
- 看参数含义、正则项设计、收敛相关说明。

2. NumPy 官方文档（向量化、广播机制）
- 解决实现时的维度与性能问题。

3. 机器学习常用指标文档（sklearn metrics）
- 明确 Precision/Recall/F1/AUC 的定义和使用场景。

## 5.3 若你要冲高分（拓展阅读）

1. 《The Elements of Statistical Learning》相关章节（广义线性模型）
2. 《Deep Learning》（Goodfellow）优化与数值计算基础章节
3. 校内往届优秀报告（如果能获取）

## 6. 我建议你的执行时间线（3-5 天）

1. Day 1
- 完成数学推导与代码骨架
- 在小样本上跑通训练，loss 明显下降

2. Day 2
- 完成完整数据实验
- 记录曲线与主要指标

3. Day 3
- 做 2-3 组对比实验（学习率、正则、阈值）
- 整理图表与结论

4. Day 4-5（可选）
- 做误差分析与可解释性分析
- 打磨报告与代码注释

## 7. 验收清单（提交前自查）

1. 代码完全基于 numpy 实现核心训练流程。
2. 训练过程可复现（固定随机种子）。
3. 提供至少一张收敛曲线。
4. 提供测试集核心指标并解释结果。
5. 报告中包含方法、实验、分析、结论四部分。

如果你愿意，我下一步可以直接在仓库里补一版最简但完整的 numpy 实现模板（含训练、评估、可视化脚手架），让你按数据集直接替换就能跑。
