# 整个任务的逻辑顺序与问题记录

## 1. 明确任务目标

本次任务的核心目标是使用 numpy 从零实现 Logistic Regression，并完成一个完整的机器学习实验。任务重点不是简单调用现成模型，而是要自己实现训练流程，并通过实验说明模型效果。

因此，项目需要覆盖以下内容：

- 数据读取与预处理
- 损失函数构建
- 梯度计算
- 参数更新
- 模型训练
- 模型评估
- 不同实验方案对比
- 结果分析与改进

## 2. 检查原始项目

一开始我们打开了 GitHub 仓库中的原始项目。项目里已经有一些 Logistic Regression 相关代码，例如乳腺癌数据集上的二分类实验、mini-batch 训练脚本、L1 正则化 Notebook 和实验报告草稿。

原项目可以体现 Logistic Regression 的基础实现，但也存在一个问题：原始数据集规模较小，样本量不足以很好地体现不同优化方法在训练时间和效果上的差异。

因此，如果直接沿用原数据集，报告内容会比较单薄，尤其是做 full-batch GD、mini-batch GD 和 SGD 的时间对比时，说服力不够强。

## 3. 重新选择数据集

为了让报告更完整，我们重新选择了数据集。考虑过的数据集包括 Breast Cancer、Heart Disease 和 Bank Marketing。

最终选择的是 UCI Bank Marketing Dataset，原因如下：

- 它是二分类任务，适合 Logistic Regression。
- 样本量较大，共 41188 条样本。
- 数据中既有数值特征，也有类别特征，适合展示数据预处理过程。
- 任务背景清楚，是预测客户是否订阅银行定期存款。
- 数据量足够，可以更明显地比较不同梯度下降方法的时间和效果。

## 4. 下载并整理数据

下载 UCI 数据集后，发现下载包中包含多个 zip 文件。我们最终使用的是新版完整数据：

```text
bank-additional-full.csv
```

该数据集的基本情况如下：

```text
样本数：41188
输入特征数：20
目标列：y
正类 yes 占比：约 11.3%
负类 no 占比：约 88.7%
```

在检查数据时发现了第一个重要问题：类别严重不平衡。大多数样本都是 `no`，真正订阅定期存款的 `yes` 样本较少。

这个问题直接影响后续模型评价。如果只看 Accuracy，模型可能看起来不错，但实际上可能无法很好地识别少数类 `yes`。

## 5. 编写基础 Logistic Regression

之后我们编写了新的主代码文件：

```text
bank_marketing_logistic_regression.py
```

基础流程包括：

1. 读取 Bank Marketing 数据集
2. 对类别特征进行 one-hot 编码
3. 对数值特征进行标准化
4. 分层划分 train / validation / test
5. 使用 numpy 实现 sigmoid 函数
6. 使用 numpy 实现二元交叉熵损失
7. 使用 numpy 实现梯度计算
8. 使用梯度下降更新参数
9. 输出 Accuracy、Precision、Recall、F1 和 AUC

这一步完成了从数据到模型训练的基础闭环。

## 6. 加入三种梯度下降方法对比

为了满足实验对比要求，我们实现了三种梯度下降方法：

| 方法 | 含义 |
|---|---|
| Full-batch GD | 每次使用全部训练集计算梯度 |
| Mini-batch GD | 每次使用一小批样本计算梯度 |
| SGD | 每次使用一个样本计算梯度 |

实验发现：

- Full-batch GD 更新稳定，但每个 epoch 中更新次数少，效果一般。
- SGD 更新非常频繁，但训练时间明显更长，波动也更大。
- Mini-batch GD 在速度和效果之间更加平衡。

因此，后续优化主要以 mini-batch 版本为基础。

## 7. 加入学习率对比

接下来我们比较了不同 learning rate 的效果，例如：

```text
0.02 / 0.05 / 0.1 / 0.2
```

学习率决定每次参数更新的步长。学习率过小会导致训练慢，学习率过大可能导致训练不稳定。

实验结果显示，不同学习率下结果有一定差异，但整体都能收敛。其中 0.05 和 0.1 的表现较稳定，因此后续实验主要围绕这些学习率展开。

## 8. 加入 L2 正则化和 Weight Decay

用户提供了一段 PyTorch 风格的代码，其中包含：

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...)
```

由于本项目要求是 numpy 手写 Logistic Regression，所以没有直接改成 PyTorch，而是实现了等价思想：

- `weight_decay` 对应 L2 正则化
- `ReduceLROnPlateau` 对应学习率调度器

在 numpy 版本中，我们加入了：

```text
l2 = 0.0001
scheduler_factor = 0.1
scheduler_patience = 5
```

这样模型在训练时不仅要降低分类损失，还会限制参数过大，从而降低过拟合风险。

## 9. 发现固定 epoch 的问题

一开始 scheduler 版本虽然加入了学习率调度，但仍然固定训练 80 个 epoch。

用户提出：如果使用 scheduler，就不应该简单固定 epoch。

因此我们进一步修改训练逻辑：

- 设置最大 epoch，例如 300
- 使用验证集 loss 判断是否进入平台期
- 如果验证集 loss 不再下降，就降低学习率
- 如果长期没有改善，触发 early stopping

这样训练流程更接近真实机器学习项目，而不是机械地训练固定轮数。

## 10. 分析评分较低的原因

初始模型的 Accuracy 接近 90%，但 F1 并不高。分析后发现，主要原因是类别不平衡。

在 Bank Marketing 数据集中：

```text
yes 约 11.3%
no 约 88.7%
```

如果模型倾向于预测 `no`，也能获得较高 Accuracy，但这并不代表模型真正能识别出会订阅的客户。

初始模型的问题主要体现在：

- Recall 偏低
- F1 偏低
- 模型对正类 `yes` 的识别不足

因此，后续优化目标不再只追求 Accuracy，而是重点提升 Recall 和 F1。

## 11. 加入阈值调优

逻辑回归输出的是概率，默认情况下使用 0.5 作为分类阈值：

```text
p >= 0.5 -> yes
p < 0.5 -> no
```

但在类别不平衡任务中，0.5 不一定合适。很多真实的 `yes` 样本可能预测概率不到 0.5，因此会被错分成 `no`。

为了解决这个问题，我们加入了验证集阈值搜索：

```text
threshold 从 0.05 到 0.95
选择 validation F1 最高的 threshold
```

加入阈值调优后，F1 明显提升，说明模型本身具有较好的概率排序能力，只是默认阈值没有充分利用这种能力。

## 12. 加入 Class Weight

为了解决类别不平衡问题，我们进一步加入了 class weight。

正类权重的计算方式是：

```text
pos_weight = 负样本数量 / 正样本数量
```

在本任务中：

```text
pos_weight ≈ 7.876
```

这意味着在计算损失和梯度时，模型会更重视正类 `yes` 样本。

加入 class weight 后，模型不再过度偏向多数类 `no`，Recall 大幅提升，F1 也进一步提升。

## 13. 改进 Scheduler 的监控指标

最开始 scheduler 是根据训练 loss 判断是否降低学习率。但训练 loss 只能说明模型对训练集拟合得怎么样，不能很好地反映泛化能力。

因此，我们把 scheduler 改成监控 validation loss。

最终 scheduler 的逻辑是：

```text
如果 validation loss 不下降
就降低学习率
如果长期不改善
就提前停止
```

这样做更符合规范的机器学习实验流程，也更能防止模型只在训练集上继续优化。

## 14. 做 Duration 特征对比

Bank Marketing 数据集中有一个重要特征：

```text
duration
```

它表示通话持续时间。

实验中我们做了两组对比：

| 实验 | 含义 |
|---|---|
| with duration | 使用 duration 特征 |
| without duration | 删除 duration 特征 |

结果显示，去掉 duration 后 F1 和 AUC 明显下降，说明通话时长对预测是否订阅有很强的信息量。

但这个特征也存在潜在问题：在真实预测场景中，duration 通常是在通话结束后才知道的。如果任务目标是在通话前预测客户是否会订阅，那么 duration 可能造成数据泄露。

因此，报告中可以同时展示两种结果，并说明 duration 的实际含义和限制。

## 15. 最终最佳模型

最终表现最好的模型是：

```text
Mini-batch + class weight + validation scheduler + threshold tuning
```

相比原始 mini-batch 模型，最终模型在 Recall 和 F1 上有明显提升。

原始 mini-batch 模型大致结果：

```text
F1 ≈ 0.51
Recall ≈ 0.42
```

最终模型大致结果：

```text
F1 ≈ 0.63
Recall ≈ 0.78
```

虽然最终模型的 Accuracy 略有下降，但这是合理的。因为模型不再只追求预测多数类，而是更重视识别少数类 `yes`。

对于当前业务任务来说，找到更多可能订阅的客户比单纯提高 Accuracy 更有意义。

## 16. 文件整理

后来我们将 Bank Marketing 相关的新代码、数据和结果整理到了独立文件夹：

```text
bank_marketing_project/
```

其中主要包括：

```text
bank_marketing_logistic_regression.py
data/
outputs/
learning_summary.md
task_workflow.md
```

输出结果包括：

```text
summary.md
gradient_descent_comparison.csv
learning_rate_comparison.csv
l2_comparison.csv
duration_comparison.csv
gd_loss_curves.svg
best_confusion_matrix.svg
```

这样做的好处是，新实验和原始项目文件分开，结构更加清晰，后续写报告或提交代码也更方便。

## 17. 遇到的工程问题

整个过程中也遇到了一些工程环境问题。

首先，Git 一开始不能直接使用，后来发现 Git 安装在：

```text
D:\Git
```

因此后续通过完整路径调用 Git。

其次，SSH 连接 GitHub 的 22 端口失败，所以改用 HTTPS 克隆仓库。

第三，WSL 和 Codex 所在的 Windows 沙箱用户不是同一个环境，因此 Codex 不能直接使用用户 WSL 中的 Git。

第四，PowerShell 读取中文文件时显示乱码，但检查后发现文件本身是 UTF-8 编码，并没有真正损坏。

第五，中途出现磁盘空间不足，导致命令执行器无法启动。后来清理了 UCI zip 中间文件、旧预测文件和 Windows Temp 中的大型临时文件，释放了 800MB 以上空间。

这些问题虽然不一定需要全部写进正式报告，但它们是项目实践中真实遇到并解决的问题。

## 18. 整体任务主线

整个任务可以总结为下面这条路线：

```text
检查原始项目
-> 发现数据集规模偏小
-> 选择 UCI Bank Marketing 数据集
-> 编写 numpy Logistic Regression
-> 完成数据预处理
-> 实现三种梯度下降方法
-> 加入学习率对比
-> 加入 L2 / weight decay
-> 加入 scheduler
-> 发现类别不平衡导致 F1 偏低
-> 加入 threshold tuning
-> 加入 class weight
-> 改用 validation loss 驱动 scheduler
-> 做 duration 特征对比
-> 得到最终模型
-> 整理代码、数据、结果和总结文档
```

这条路线体现了一个机器学习项目从基础实现到逐步改进的完整过程。每一次修改都不是随意增加功能，而是针对前一步实验中发现的问题进行改进。

## 19. 总结

本任务最终不仅完成了 Logistic Regression 的 numpy 实现，还完成了多个实验对比和模型改进。

最重要的收获是：机器学习实验不是一次写完代码就结束，而是需要根据数据特点和评价结果不断分析问题、提出改进、重新实验。

本项目中，最关键的问题是类别不平衡。围绕这个问题，我们加入了阈值调优和 class weight，最终显著提升了模型对正类客户的识别能力。

因此，这个任务的完整逻辑可以概括为：

```text
先实现基础模型，再通过实验发现问题，最后针对问题逐步优化。
```
