# Bank Marketing 大数据集上的 8 种损失函数实验

## 1. 实验目的

之前我们在乳腺癌小数据集上测试了 8 种不同损失函数。结果显示，在小而干净的数据集上，很多自定义损失函数都能取得不错效果，甚至有些函数在一次划分中达到满分。

但是这并不能说明这些损失函数在更大、更复杂的数据集上同样可靠。为了验证这一点，本实验将同样的 8 种损失函数放到 Bank Marketing 数据集上进行比较。

Bank Marketing 数据集有 41188 条样本，包含数值特征和类别特征，目标是预测客户是否会订阅银行定期存款。相比乳腺癌数据集，它具有以下特点：

- 样本量更大
- 特征类型更复杂
- 类别不平衡更明显
- 数据噪声更多
- 任务本身更难

因此，它更适合观察不同损失函数在真实复杂数据上的稳定性和泛化效果。

## 2. 实验设置

模型仍然固定为二分类 Logistic Regression：

```text
z = w^T x + b
p = sigmoid(z) = 1 / (1 + exp(-z))
```

所有实验只替换损失函数，不改变模型结构。

数据处理方式：

- 类别特征使用 one-hot 编码
- 数值特征使用标准化
- 使用分层划分 train / validation / test
- 使用 class weight 处理类别不平衡
- 在验证集上搜索最佳分类阈值

为了分析 `duration` 特征的影响，本实验做了两组：

| 设置 | 含义 |
|---|---|
| with duration | 使用全部特征，包括通话时长 |
| without duration | 删除 duration 特征 |

## 3. 8 个损失函数的数学形式

下面统一记：

```text
z = w^T x + b
p = sigmoid(z)
y ∈ {0, 1}
y' = 2y - 1 ∈ {-1, 1}
margin = y' z
```

其中，`p` 是模型预测为正类的概率，`margin` 表示分类间隔。

### 3.1 Logistic BCE

数学形式：

```text
L = -[y log(p) + (1-y) log(1-p)]
```

这是标准 Logistic Regression 使用的二元交叉熵损失。

它的梯度形式非常简洁：

```text
dL/dz = p - y
```

优点：

- 概率解释清楚
- 梯度性质好
- 对自信但错误的预测仍有较强修正能力
- 优化稳定
- 泛化表现通常较可靠

### 3.2 Linear Probability Loss

数学形式：

```text
L = y(1-p) + (1-y)p
```

如果真实标签是 1，则损失为 `1-p`；如果真实标签是 0，则损失为 `p`。

它符合“预测错了损失变大”的直觉。但是它的问题在于梯度会受到 sigmoid 饱和影响。

它的梯度中包含：

```text
p(1-p)
```

当模型非常自信时，`p` 接近 0 或 1，`p(1-p)` 接近 0。此时即使模型预测错了，梯度也可能变小。

### 3.3 Squared Probability Loss

数学形式：

```text
L = 0.5(p-y)^2
```

这是类似均方误差的概率损失。它直接惩罚预测概率和真实标签之间的距离。

优点是形式简单，直观易懂。缺点是梯度同样受到 `p(1-p)` 的影响，在 sigmoid 饱和区可能更新不足。

### 3.4 Quartic Probability Loss

数学形式：

```text
L = 0.25(p-y)^4
```

这是四次多项式概率损失。相比平方损失，它对较大误差有更强惩罚。

但它也有两个问题：

- 小误差区域梯度可能更弱
- 仍然会受到 sigmoid 饱和影响

因此，它在小数据集上可能表现不错，但在复杂数据上不一定稳定。

### 3.5 Exponential Margin Loss

数学形式：

```text
L = exp(-margin)
```

其中：

```text
margin = y' z
```

如果分类正确且 margin 大，损失接近 0；如果分类错误，margin 为负，损失会指数级增大。

优点是对错误样本惩罚强。缺点是对异常点和噪声比较敏感，容易出现梯度过大，因此实际代码中需要对 margin 做裁剪。

### 3.6 Squared Hinge Margin Loss

数学形式：

```text
L = max(0, 1 - margin)^2
```

这个损失关注分类间隔。如果样本不仅分对，而且 margin 大于 1，损失就是 0。

它更接近支持向量机思想，适合分类边界学习。但它不直接优化概率，因此概率解释不如 Logistic BCE。

### 3.7 Cubic Hinge Margin Loss

数学形式：

```text
L = max(0, 1 - margin)^3
```

这是 hinge loss 的三次版本。它对 margin 不足的样本惩罚更强。

优点是能更强调分类困难的样本。缺点是高阶惩罚可能导致训练更敏感，因此需要更小的学习率。

### 3.8 Log-cosh Margin Loss

数学形式：

```text
L = log(cosh(1 - margin))
```

它是一个平滑的 margin 型损失。相比 hinge，它更光滑；相比 exponential，它增长没有那么激烈。

它的设计思想是在错误时产生较大损失，同时避免指数损失那样过于敏感。

## 4. 实验结果

完整结果保存在：

```text
outputs/custom_losses/bank_custom_loss_comparison.csv
outputs/custom_losses/bank_custom_loss_results_summary.md
```

### 4.1 使用 duration 特征

| Loss | Threshold | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic BCE | 0.730000 | 0.904964 | 0.560467 | 0.724138 | 0.631876 | 0.935755 |
| Linear probability loss | 0.950000 | 0.879354 | 0.479630 | 0.837284 | 0.609890 | 0.932307 |
| Squared probability loss | 0.780000 | 0.896711 | 0.528164 | 0.778017 | 0.629194 | 0.933463 |
| Quartic probability loss | 0.570000 | 0.900959 | 0.544094 | 0.744612 | 0.628753 | 0.935933 |
| Exponential margin loss | 0.570000 | 0.891370 | 0.513971 | 0.654095 | 0.575628 | 0.918467 |
| Squared hinge margin loss | 0.590000 | 0.904964 | 0.563208 | 0.696121 | 0.622651 | 0.935707 |
| Cubic hinge margin loss | 0.540000 | 0.898167 | 0.536208 | 0.710129 | 0.611034 | 0.933866 |
| Log-cosh margin loss | 0.560000 | 0.889671 | 0.507370 | 0.704741 | 0.589986 | 0.932887 |

在使用 `duration` 的情况下，Logistic BCE 的 F1 最高，达到 0.631876。Squared probability 和 Quartic probability 也比较接近，但整体上 Logistic BCE 更均衡。

Linear probability loss 的 Recall 很高，但 Precision 明显较低，说明它更倾向于预测正类，误报较多。

Exponential margin loss 的表现相对较弱，可能是因为它对难样本和噪声比较敏感。

### 4.2 删除 duration 特征

| Loss | Threshold | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic BCE | 0.650000 | 0.871222 | 0.442023 | 0.546336 | 0.488675 | 0.795258 |
| Linear probability loss | 0.950000 | 0.866610 | 0.430544 | 0.571121 | 0.490968 | 0.784796 |
| Squared probability loss | 0.630000 | 0.868552 | 0.436527 | 0.574353 | 0.496045 | 0.794004 |
| Quartic probability loss | 0.550000 | 0.870979 | 0.441355 | 0.547414 | 0.488696 | 0.794742 |
| Exponential margin loss | 0.590000 | 0.873771 | 0.449275 | 0.534483 | 0.488189 | 0.794342 |
| Squared hinge margin loss | 0.570000 | 0.872315 | 0.444544 | 0.535560 | 0.485826 | 0.794884 |
| Cubic hinge margin loss | 0.540000 | 0.872679 | 0.445934 | 0.537716 | 0.487543 | 0.795615 |
| Log-cosh margin loss | 0.600000 | 0.871465 | 0.440183 | 0.519397 | 0.476520 | 0.791979 |

删除 `duration` 后，所有损失函数的表现都明显下降。最好的 F1 是 Squared probability loss，约为 0.496045，但它只比 Logistic BCE 略高。

这说明在没有 `duration` 的情况下，任务本身变难了，损失函数之间的差异变小，整体上都难以达到使用 `duration` 时的效果。

## 5. 和乳腺癌小数据集的对比

乳腺癌数据集上，很多自定义 loss 都能达到非常高的分数。原因是：

- 数据量小
- 特征区分度强
- 噪声相对较少
- 线性模型已经足够有效

Bank Marketing 数据集上，结果明显不同：

- 自定义 loss 之间差异更明显
- 某些 loss 的 Recall 高但 Precision 低
- 某些 margin loss 对噪声更敏感
- Logistic BCE 的整体表现更稳

这说明，小数据集上可行的自定义 loss，不一定能在大数据集上继续保持优势。

## 6. 为什么 Logistic BCE 在大数据集上更稳

### 6.1 梯度更适合概率模型

Logistic BCE 的梯度是：

```text
dL/dz = p - y
```

它不会额外乘上 `p(1-p)`，因此在模型预测非常错误时，仍然能保持有效梯度。

而 Linear probability、Squared probability 和 Quartic probability 的梯度都会受到 sigmoid 饱和影响。大数据集中样本更多、噪声更多，这类梯度问题更容易暴露。

### 6.2 概率解释更好

Bank Marketing 是一个真实业务预测问题。模型输出的概率不仅用于分类，还可以理解为客户订阅的可能性。

Logistic BCE 来自伯努利分布的最大似然估计，因此概率解释更自然。其他自定义损失虽然可能分类效果不错，但概率校准不一定可靠。

### 6.3 对复杂数据更稳定

Bank Marketing 中存在类别不平衡和多种类别特征。Logistic BCE 在这种情况下仍然取得了较高 AUC 和 F1，说明它对复杂数据更稳。

一些自定义 loss 可能在某个指标上表现突出，例如 Linear probability 的 Recall 高，但 Precision 较低，说明它可能牺牲了误报率。

## 7. 结果解读

在 with duration 设置下，Logistic BCE 的 F1 最高：

```text
F1 = 0.631876
AUC = 0.935755
```

Squared probability 和 Quartic probability 的 F1 也很接近，说明这些自定义概率型损失在这个任务上并非完全不可用。

但是，Linear probability 虽然 Recall 高达 0.837284，但 Precision 只有 0.479630。它找出了更多正类客户，但误报也更多。如果业务目标是尽量覆盖潜在客户，它可能有一定价值；如果希望控制误报，它就不如 Logistic BCE 均衡。

在 without duration 设置下，所有 loss 都明显下降。这再次说明 `duration` 是强特征，也说明数据本身比乳腺癌数据集更难。

## 8. 大数据集上自定义 loss 是否可行？

结论是：可以尝试，但不能只凭直觉设计。

在大数据集上，自定义 loss 需要考虑：

- 梯度是否稳定
- 是否容易出现梯度消失或梯度爆炸
- 是否对噪声过于敏感
- 是否适合类别不平衡
- 输出概率是否有解释意义
- 是否能在验证集和测试集上稳定泛化

本实验中，自定义 loss 确实可以训练出可用模型，但 Logistic BCE 在整体表现上仍然最稳。

## 9. Epoch 翻倍后的进一步验证

在第一次大数据集实验中，有一个值得注意的现象：某些自定义损失函数在部分指标上接近甚至超过 Logistic BCE。例如，在删除 `duration` 特征的设置下，Linear probability loss 的 F1 曾略高于 Logistic BCE。

这引出了一个合理怀疑：

```text
是不是 Linear probability loss 收敛更快，所以在固定 epoch 下看起来更好？
如果把训练 epoch 翻倍，Logistic BCE 会不会追上或超过它？
```

为了验证这个问题，我们把 8 个损失函数的训练 epoch 从 80 翻倍到 160，并重新运行 Bank Marketing 实验。

### 9.1 使用 duration 特征，160 epoch

| Loss | Threshold | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic BCE | 0.710000 | 0.901202 | 0.545600 | 0.734914 | 0.626263 | 0.936083 |
| Linear probability loss | 0.950000 | 0.875713 | 0.471360 | 0.851293 | 0.606759 | 0.931807 |
| Squared probability loss | 0.790000 | 0.895983 | 0.525969 | 0.774784 | 0.626580 | 0.933325 |
| Quartic probability loss | 0.570000 | 0.899866 | 0.540329 | 0.743534 | 0.625850 | 0.936009 |
| Exponential margin loss | 0.570000 | 0.891734 | 0.515228 | 0.656250 | 0.577251 | 0.918911 |
| Squared hinge margin loss | 0.590000 | 0.902901 | 0.555556 | 0.689655 | 0.615385 | 0.935564 |
| Cubic hinge margin loss | 0.540000 | 0.898653 | 0.537836 | 0.712284 | 0.612888 | 0.934423 |
| Log-cosh margin loss | 0.540000 | 0.884695 | 0.492350 | 0.762931 | 0.598478 | 0.931806 |

在使用 `duration` 的情况下，epoch 翻倍后 Logistic BCE 的 F1 为 0.626263，而 Linear probability loss 的 F1 为 0.606759。Linear probability loss 的 Recall 更高，但 Precision 明显更低，说明它更倾向于预测正类，误报较多。

因此，在这个设置下，Linear probability loss 并不是因为收敛更快才更优；它实际上没有超过 Logistic BCE 的综合 F1。

### 9.2 删除 duration 特征，160 epoch

| Loss | Threshold | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic BCE | 0.610000 | 0.866974 | 0.433014 | 0.585129 | 0.497709 | 0.793168 |
| Linear probability loss | 0.950000 | 0.866489 | 0.431090 | 0.579741 | 0.494485 | 0.782433 |
| Squared probability loss | 0.630000 | 0.869280 | 0.438277 | 0.570043 | 0.495550 | 0.793416 |
| Quartic probability loss | 0.540000 | 0.868795 | 0.437653 | 0.578664 | 0.498376 | 0.792121 |
| Exponential margin loss | 0.590000 | 0.874014 | 0.450091 | 0.534483 | 0.488670 | 0.794640 |
| Squared hinge margin loss | 0.550000 | 0.866003 | 0.430490 | 0.587284 | 0.496809 | 0.793155 |
| Cubic hinge margin loss | 0.530000 | 0.866974 | 0.432584 | 0.580819 | 0.495860 | 0.794653 |
| Log-cosh margin loss | 0.590000 | 0.871101 | 0.441536 | 0.545259 | 0.487946 | 0.792028 |

删除 `duration` 后，所有损失函数的效果都下降。160 epoch 下，Logistic BCE 的 F1 为 0.497709，Linear probability loss 的 F1 为 0.494485。也就是说，在训练更久后，Logistic BCE 反而略微超过 Linear probability loss。

这说明第一次实验中 Linear probability loss 在 without duration 设置下略高，很可能不是因为它本质更优，而是受训练轮数、阈值搜索、数据划分和优化路径共同影响。

### 9.3 对“线性损失收敛更快”的判断

从 160 epoch 的结果看，Linear probability loss 的特点更像是：

```text
Recall 较高，但 Precision 较低。
```

它更容易把样本判成正类，因此能找出更多真正的 `yes`，但同时也带来更多误报。

而 Logistic BCE 更均衡，AUC 也通常更高。这说明它不只是收敛速度问题，而是损失函数本身对概率和梯度的处理更稳定。

因此，本次 epoch 翻倍实验支持下面的结论：

```text
Linear probability loss 可以作为可行的自定义损失，但它并没有在大数据集上稳定超过 Logistic BCE。
Logistic BCE 的综合表现仍然更稳，尤其在训练轮数增加后更明显。
```

## 10. 不同初始点下的稳定性验证

由于 Linear probability loss 不是全局凸函数，一个自然的怀疑是：它在零初始化时表现不错，但如果换成不同随机初始点，缺点可能会暴露出来。

为验证这一点，我们额外做了初始化稳定性实验，只比较两种损失：

```text
Logistic BCE
Linear probability loss
```

实验设置：

- 初始化尺度 `init_scale = 0, 0.1, 1.0, 3.0`
- 每个尺度运行 3 个随机种子
- 训练 epoch 为 60
- 评价指标为测试集 F1、AUC 和 Accuracy

结果如下：

| Loss | Init scale | Mean F1 | Std F1 | Min F1 | Max F1 | Mean AUC |
|---|---:|---:|---:|---:|---:|---:|
| Linear probability loss | 0.0 | 0.611927 | 0.000388 | 0.611551 | 0.612326 | 0.932446 |
| Linear probability loss | 0.1 | 0.611900 | 0.000940 | 0.610935 | 0.612813 | 0.932495 |
| Linear probability loss | 1.0 | 0.602927 | 0.004601 | 0.598844 | 0.607912 | 0.930890 |
| Linear probability loss | 3.0 | 0.448410 | 0.024596 | 0.420016 | 0.463146 | 0.876027 |
| Logistic BCE | 0.0 | 0.631040 | 0.003482 | 0.628046 | 0.634861 | 0.935574 |
| Logistic BCE | 0.1 | 0.630376 | 0.004499 | 0.625863 | 0.634861 | 0.935578 |
| Logistic BCE | 1.0 | 0.629070 | 0.004413 | 0.626143 | 0.634146 | 0.935486 |
| Logistic BCE | 3.0 | 0.621605 | 0.007151 | 0.613372 | 0.626272 | 0.934208 |

这个结果很清楚：

```text
Linear probability loss 对初始化更敏感。
Logistic BCE 对初始化更稳定。
```

当初始化尺度较小时，Linear probability loss 还能保持较好效果；但当初始化尺度增大到 3.0 时，它的平均 F1 从约 0.612 降到约 0.448，AUC 也从约 0.932 降到约 0.876。

相比之下，Logistic BCE 在同样的初始化变化下仍然比较稳定。即使 `init_scale = 3.0`，它的平均 F1 仍然有约 0.622，AUC 仍然约为 0.934。

### 10.1 数学原因

Linear probability loss 可以写成 margin 形式：

```text
L_linear(m) = sigmoid(-m)
```

它的导数是：

```text
dL_linear/dm = -sigmoid(m)sigmoid(-m)
```

当模型初始化较大时，很多样本的 `|m|` 会很大。如果某些样本一开始就被非常自信地分错，则：

```text
m << 0
```

此时：

```text
sigmoid(m) ≈ 0
sigmoid(-m) ≈ 1
dL_linear/dm ≈ 0
```

也就是说，模型虽然错得很离谱，但梯度反而接近 0，难以修正。

Logistic BCE 的 margin 形式是：

```text
L_logistic(m) = log(1 + exp(-m))
```

它的导数是：

```text
dL_logistic/dm = -1 / (1 + exp(m))
```

当 `m << 0` 时：

```text
dL_logistic/dm ≈ -1
```

也就是说，如果模型非常自信地预测错了，Logistic BCE 仍然会给出很强的梯度，把模型往正确方向拉回来。

这就是 Logistic BCE 比 Linear probability loss 更稳定的核心数学原因。

### 10.2 结论

你的判断是正确的：

```text
Linear probability loss 的非凸性和梯度饱和问题，在不同初始点下会暴露出来。
```

它在零初始化或小初始化时可以表现不错，但一旦初始点较远，模型可能进入梯度很弱的区域，导致优化困难。

Logistic BCE 的优势不只是凸性，还包括它在错误且自信时仍然保持有效梯度。因此它在不同初始点下更可靠。

## 11. 最终结论

“预测错了损失变大”是设计损失函数的必要条件之一，但远远不够。

一个真正好用的损失函数还需要：

- 错误时有合适的梯度
- 正确时能逐渐减小更新
- 优化过程稳定
- 对噪声不过度敏感
- 能输出合理概率
- 在小数据和大数据上都能泛化

乳腺癌小数据集说明：自定义 loss 在简单任务上可能表现很好。

Bank Marketing 大数据集说明：数据一复杂，Logistic BCE 的优势会更明显。它不是因为“公式经典”才好，而是因为它的梯度、概率解释和优化性质都更适合二分类 Logistic Regression。

因此，本实验的最终观点是：

```text
自定义损失函数可以作为探索和对比实验；
但在真实任务和大数据集上，Logistic BCE 仍然是更可靠的默认选择。
```

## 12. 本实验生成的文件

实验代码：

```text
bank_custom_loss_experiments.py
```

实验结果：

```text
outputs/custom_losses/bank_custom_loss_comparison.csv
outputs/custom_losses/bank_custom_loss_results_summary.md
outputs/custom_losses/bank_custom_loss_with_duration.svg
outputs/custom_losses/bank_custom_loss_without_duration.svg
```
