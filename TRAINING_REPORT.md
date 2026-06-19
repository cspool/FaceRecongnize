# 人脸识别实验报告

## 1. 任务要求与实验目标

`task.md` 的要求是在人脸识别任务上应用模式识别方法，提取不同的人脸特征进行识别，识别方法不限，并对识别率进行比较。给定数据包括 PIE 数据库和 Essex Face Recognition Data。

本报告围绕两个目标组织实验：

1. 在 PIE 官方划分上完成标准 closed-set 身份分类，作为基础识别结果。
2. 在 Essex FRD 上设计更严格的 identity-disjoint 协议，避免训练和测试身份重叠，并重点考察最难的 `faces96` 与 `grimace` 子集。

为了避免只报告过高但不可靠的指标，FRD 部分同时加入随机权重和原始像素特征的 sanity baseline。最终结果表只保留关键指标：FRD 使用 `TAR@FAR=0.1%`、`EER` 和 `1-shot Top-1`，PIE 使用 closed-set `Top-1`。

## 2. 数据集

### 2.1 PIE 数据库

PIE 数据库包含 68 个人的人脸图像，每人有不同姿态、光照和表情。作业数据中提供 5 种姿态，并包含 `isTest` 字段。

本实验使用 `isTest=1` 的样本作为测试集，不参与训练。`isTest=0` 的样本再划分训练集和验证集。

### 2.2 Essex Face Recognition Data

Essex FRD 包含 395 个身份，约每人 20 张图像，分为 4 个子集：

- `faces94`
- `faces95`
- `faces96`
- `grimace`

其中 `faces96` 和 `grimace` 更难，主要因为背景、尺度、人脸表情和采集条件变化更明显。因此，FRD 的严格实验把 `faces96` 与 `grimace` 作为验证/测试核心子集。

## 3. 术语与缩写说明

| 缩写 / 术语 | 全称或含义 | 本报告中的具体含义 |
| --- | --- | --- |
| PIE | Pose, Illumination and Expression | 人脸数据集，强调姿态、光照和表情变化 |
| FRD | Face Recognition Data | University of Essex Face Recognition Data 数据集 |
| ResNet18 | 18-layer Residual Network | 18 层残差卷积神经网络，本实验的主干模型 |
| ImageNet | ImageNet large-scale visual dataset | ResNet18 预训练权重来源 |
| embedding | 特征嵌入向量 | 模型倒数第二层输出的 512 维人脸特征 |
| avgpool | average pooling | ResNet18 全局平均池化层，输出 512 维特征 |
| L2 归一化 | L2 normalization | 将 embedding 归一化到单位长度，方便用 cosine 相似度比较 |
| closed-set | 闭集识别 | 训练和测试身份集合相同，测试图像一定属于训练过的身份 |
| identity-disjoint | 身份不重叠 | 训练、验证、测试身份互不相同，用于检验未见身份泛化能力 |
| verification | 人脸验证 | 判断两张图像是否属于同一身份 |
| 1-shot identification | 单样本库识别 | 每个测试身份只放 1 张 gallery 图像，其余图像作为 probe 查询 |
| Top-1 | 第一名命中率 | 模型最相似的第 1 个身份是否正确 |
| ROC-AUC / AUC | Receiver Operating Characteristic - Area Under Curve | 排序能力指标，本实验仅用于选择 checkpoint 和辅助解释，不作为最终主指标 |
| FAR | False Accept Rate | 误接收率，把不同人错误判断为同一人的比例 |
| TAR | True Accept Rate | 真接收率，把同一人正确判断为同一人的比例 |
| TAR@FAR=0.1% | TAR under FAR constrained to 0.1% | 当误接收率约束到 0.1% 时的真接收率，是 FRD 最严格主指标 |
| EER | Equal Error Rate | 误接收率和误拒绝率相等时的错误率，越低越好 |

## 4. 特征与识别方法

本实验比较了三类人脸特征。

| 特征 | 说明 | 用途 |
| --- | --- | --- |
| 微调 ResNet18 特征 | 使用 ImageNet 预训练 ResNet18，替换分类头后训练；FRD 评估时取 avgpool 后 512 维向量并 L2 归一化 | 主模型 |
| 随机 ResNet18 特征 | ResNet18 不加载 ImageNet、不训练，直接抽取 512 维随机卷积特征 | 检查协议是否过于容易 |
| 32x32 原始像素特征 | 图像下采样后展开并归一化，用 cosine 相似度比较 | 检查低层视觉相似性 |

识别方法分为两类。

PIE 使用 closed-set 分类：训练和测试身份集合相同，模型输出 68 类身份概率，报告 Top-1。

FRD 使用 identity-disjoint embedding 评估：训练身份、验证身份、测试身份互不重叠。训练阶段只用训练身份的分类头学习特征；测试阶段丢弃分类头，使用归一化 embedding 做 verification 和 1-shot identification。

这里的“类别”指 person identity。`train_fractions` 和 `hard_train_fractions` 选择的是各数据子集中的身份比例，不是图像比例。未被选入训练的 easy 身份会被排除，不会移动到验证/测试；验证和测试只来自 hard 子集，并且身份不和训练集重叠。

## 5. 训练与划分设置

### 5.1 通用训练设置

- Backbone: ResNet18
- 输入尺寸: 224x224
- 优化器: AdamW
- 数据增强: random horizontal flip、small rotation、color jitter
- 主 FRD 训练轮数: 12 epoch
- Batch size: 256
- Backbone learning rate: `5e-5`
- Head learning rate: `5e-4`
- Weight decay: `1e-4`
- 随机种子: 42

### 5.2 FRD strict 协议

FRD strict 协议的关键约束是：

- 训练身份和测试身份不重叠。
- 验证身份和测试身份不重叠。
- 测试集不包含 `faces94` 或 `faces95`。
- 验证/测试集中只使用 hard 子集：`faces96` 和 `grimace`。
- 最佳 checkpoint 根据验证集 verification AUC 选择，再在测试集上报告结果。
- FAR 阈值先在验证集上确定，再应用到测试集。

## 6. FRD 主要实验配置

| 实验 | 训练身份 | 验证身份 | 测试身份 | 训练子集 | 验证/测试子集 | 说明 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Random weights baseline | 76 | 84 | 85 | `faces94:76` | `faces96`, `grimace` | 随机 ResNet18，不训练 |
| Half Face94 only | 76 | 84 | 85 | `faces94:76` | `faces96`, `grimace` | 最新保守主实验 |
| Half easy no face96 | 112 | 84 | 85 | `faces94:76`, `faces95:36` | `faces96`, `grimace` | 使用一半 easy 身份 |
| No face96 training | 225 | 84 | 85 | `faces94:153`, `faces95:72` | `faces96`, `grimace` | 不训练 `faces96/grimace` |
| No hard train | 225 | 83 | 85 | `faces94:153`, `faces95:72` | `faces96`, `grimace` | `--no-hard-train` |
| Limited hard train | 255 | 70 | 69 | `faces94:153`, `faces95:72`, `faces96:30` | `faces96`, `grimace` | 少量 `faces96` 参与训练 |
| Hard subject-disjoint | 326 | 34 | 34 | `faces94`, `faces95`, 部分 `faces96/grimace` | `faces96`, `grimace` | 身份不重叠，但 hard 子集也参与训练 |

最新保守主实验 `Half Face94 only` 的具体 split 为：

| Split | Identities | Images | Subsets |
| --- | ---: | ---: | --- |
| Train | 76 | 1,391 | `faces94` |
| Validation | 84 | 1,652 | `faces96`, `grimace` |
| Test | 85 | 1,672 | `faces96`, `grimace` |
| Excluded | 149 | 2,750 | `faces94`, `faces95` |

该 split 中 `faces94/faces95` 在验证和测试中均为 0，训练、验证、测试身份交集均为空。

## 7. FRD 实验结果

### 7.1 关键指标结果

| 实验 | EER | TAR@FAR=0.1% | 1-shot Top-1 | 说明 |
| --- | ---: | ---: | ---: | --- |
| Random weights baseline | 7.67% | 52.13% | 87.90% | 随机 ResNet18，不训练 |
| Half Face94 only | 9.63% | 55.35% | 87.02% | 最新保守主实验 |
| Half easy no face96 | 8.79% | 52.06% | 87.84% | 使用一半 easy 身份 |
| No face96 training | 6.69% | 64.33% | 86.70% | 全部 easy 身份训练 |
| No hard train | 6.71% | 64.10% | 86.83% | 全部 easy 身份训练，hard 仅验证/测试 |
| Limited hard train | 2.77% | 80.23% | 93.87% | 少量 `faces96` 参与训练 |
| Hard subject-disjoint | 1.17% | 84.49% | 98.74% | hard 子集也有部分身份参与训练 |

### 7.2 最新主实验按子集统计

`Half Face94 only` 的 1-shot identification 子集结果如下：

| Test subset | Probe images | 1-shot Top-1 |
| --- | ---: | ---: |
| `faces96` | 1,417 | 85.46% |
| `grimace` | 170 | 100.00% |

`grimace` 子集样本数较少，且同一子集内部视觉规律较强，因此 100% 不应单独过度解读。

## 8. PIE 与 closed-set 参考结果

PIE 使用官方 `isTest` 测试字段，属于 closed-set 身份分类。

| 数据集 / 实验 | 协议 | Top-1 | 说明 |
| --- | --- | ---: | --- |
| PIE | 官方 `isTest` 测试集 | 100.00% | 68 类身份分类 |
| FRD early closed-set | 同一身份出现在训练和测试中 | 99.74% | 仅作参考，不作为主结论 |

这些 closed-set 结果说明 ResNet18 能完成给定身份集合内的分类任务，但它们不能证明模型能识别训练中从未见过的新身份。因此 FRD 主结论以 identity-disjoint strict 协议为准。

## 9. 指标解释

### 9.1 本报告保留的关键指标

FRD 结果表只保留 3 个关键指标：

- `TAR@FAR=0.1%`：最严格主指标，表示误接收率约束到 0.1% 时的真接收率，越高越好。
- `EER`：整体阈值难度指标，越低越好。
- `1-shot Top-1`：单样本库身份检索指标，越高越好。

PIE closed-set 分类只保留 `Top-1`，因为它直接表示测试图像分类到正确身份的比例。

### 9.2 为什么不把 ROC-AUC 作为主结果

ROC-AUC 衡量同人 pair 的相似度是否整体高于异人 pair，是排序指标。它不固定实际使用阈值，因此在负样本很多、背景/姿态/采集条件规律较强的数据中可能显得很高。本实验仍用验证集 AUC 选择 checkpoint，但最终结果表不再报告 AUC。

1-shot Top-1 是在测试身份集合已知的情况下，从 gallery 中找最相似身份。它适合说明检索排序能力，但不等价于开放场景下的人脸验证，所以它是辅助关键指标，主结论仍看 `TAR@FAR=0.1%`。

### 9.3 更严格的主指标：TAR@FAR=0.1%

`TAR@FAR=0.1%` 表示在误接收率约束到 0.1% 时，真实同人样本中有多少能被接受。这个指标更接近安全要求较高的人脸验证场景。

以最新 `Half Face94 only` 为例：

- EER: 9.63%
- 1-shot Top-1: 87.02%
- TAR@FAR=0.1%: 55.35%

因此，虽然 AUC 和 Top-1 看起来较高，但在严格低误接收阈值下，模型只能召回约一半真实匹配。这个结论比“97% 准确率”更准确。

## 10. Sanity baseline 与结果合理性

随机权重 baseline 使用同一个 split，但 ResNet18 不加载 ImageNet 权重、不训练。检查结果显示：

- `random_model.pt` 与 seed=42 的新随机初始化 ResNet18 权重一致。
- 它不等于 ImageNet 权重，也不等于训练好的 checkpoint。
- `metrics.json` 中 `best_epoch=null`，说明没有训练。

随机权重结果仍然较高。这里同样只保留关键指标：

| 特征 / 检查 | TAR@FAR=0.1% | 1-shot Top-1 | 说明 |
| --- | ---: | ---: | --- |
| 随机 ResNet18 特征 | 52.13% | 87.90% | 不训练、不加载 ImageNet |
| 微调 ResNet18 特征 | 55.35% | 87.02% | 最新 `Half Face94 only` 主实验 |

额外 sanity 检查显示，原始像素特征也能得到很高的非主指标，而打乱标签或使用 Gaussian 随机特征会回到随机水平。这说明高分不是因为误加载训练权重，而是因为当前 FRD split 中存在强低层视觉相似性，例如采集条件、背景、尺度、姿态等规律。随机卷积特征和原始像素都能利用这些规律。

因此，当前实验结果是合理的，但必须谨慎解释：

- `TAR@FAR=0.1%` 才是主指标。
- 最新保守主实验的严格指标是 55.35%，并不夸张。
- 随机权重已有 52.13%，说明模型训练带来的增益有限。
- 训练中加入更多 easy 身份后，`TAR@FAR=0.1%` 提升到约 64%。
- 少量 hard 身份参与训练后，`TAR@FAR=0.1%` 提升到 80% 以上，符合训练分布更接近测试分布时性能提高的预期。

## 11. 结论

本实验完成了 task.md 要求的人脸识别任务，并比较了不同特征与不同识别协议。

PIE closed-set 分类达到 100% Top-1，说明在官方测试划分下，ResNet18 可以很好地区分已知身份。

FRD 上，identity-disjoint strict 协议更能检验泛化能力。最新保守实验只使用 50% `faces94` 身份训练，不使用 `faces95/faces96/grimace` 训练身份，测试只在未见过的 `faces96/grimace` 身份上进行。该设置下：

- EER 为 9.63%。
- 1-shot Top-1 为 87.02%。
- 更严格的 `TAR@FAR=0.1%` 为 55.35%。

最终应将结论表述为：

> 在 identity-disjoint hard-subset 协议下，模型在严格 FAR=0.1% 阈值下达到约 55% TAR；但随机权重和原始像素 baseline 也很高，说明 FRD 当前划分仍包含明显低层视觉相似性。因此结果可以说明模型具备一定识别能力，但不能单独证明其学到了鲁棒、可泛化的人脸身份特征。

如果需要报告更强性能，可以同时给出 `No hard train` 或 `Limited hard train` 结果；但应明确这些设置使用了更多训练身份，或者让部分 hard 子集参与训练，不能和 `Half Face94 only` 的保守设置直接等价。

## 12. 可复现文件

- PIE 训练脚本: `train_face_recognition.py`
- FRD closed-set 训练脚本: `train_essex_recognition.py`
- FRD strict embedding 脚本: `train_essex_strict.py`
- PIE 结果: `artifacts/pie_resnet18/metrics.json`
- FRD 最新主实验: `artifacts/essex_strict_half_face94_only_resnet18/metrics.json`
- FRD 随机权重 baseline: `artifacts/essex_strict_random_weights_half_face94_split/metrics.json`
- FRD 全 easy 训练 strict 结果: `artifacts/essex_strict_no_hard_train_resnet18/metrics.json`
- FRD limited hard 训练结果: `artifacts/essex_strict_limited_hard_train_resnet18/metrics.json`
