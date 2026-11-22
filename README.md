# microRNA-gene关系预测模型

基于深度学习的microRNA和gene关系对预测项目，使用神经网络模型预测miRNA与gene之间的调控关系。

## 项目简介

microRNA（miRNA）是一类重要的非编码RNA，通过调控基因表达参与多种生物过程。本项目利用深度学习技术，基于miRNA和gene的序列信息，预测它们之间是否存在功能性调控关系（Functional MTI）。

## 数据集

项目使用的数据集包含：
- **gene_seq.csv**：基因序列数据，包含16127个基因的序列信息
- **mirna_seq.csv**：miRNA序列数据，包含2656个miRNA的序列信息  
- **Train_test.csv**：关系对数据集，包含738对miRNA-gene关系（标签：Functional MTI / Non-Functional MTI）

### 数据格式说明

**gene_seq.csv**：
| 列名 | 说明 |
|------|------|
| label | 基因名称 |
| sequence | 基因DNA序列（ATCG） |

**mirna_seq.csv**：
| 列名 | 说明 |
|------|------|
| mirna | miRNA名称 |
| seq | miRNA序列（AUCG） |

**Train_test.csv**：
| 列名 | 说明 |
|------|------|
| miRNA | miRNA名称 |
| gene | 基因名称 |
| label | 关系标签（Functional MTI / Non-Functional MTI） |

## 技术方案

### 特征提取
- **k-mer特征**：提取2-mer和3-mer核苷酸频率特征
- **GC含量**：计算序列中G和C的比例
- **序列长度**：作为补充特征

### 模型架构
```
输入层（特征向量）
    ↓
Dense(256, ReLU) + BatchNormalization + Dropout(0.3)
    ↓
Dense(128, ReLU) + BatchNormalization + Dropout(0.3)
    ↓
Dense(64, ReLU) + BatchNormalization + Dropout(0.2)
    ↓
Dense(32, ReLU)
    ↓
Dense(1, Sigmoid) → 输出分类概率
```

### 优化策略
- **SMOTE过采样**：处理类别不平衡问题
- **早停机制**：防止模型过拟合
- **学习率调度**：自适应调整学习率
- **阈值寻优**：寻找最优分类阈值最大化F1得分

## 环境要求

### 硬件要求
| 配置 | 推荐配置 | 最低配置 |
|------|----------|----------|
| CPU | Intel Core i7-8代及以上 | Intel Core i5-6代及以上 |
| 内存 | 16GB DDR4 | 8GB DDR4 |
| GPU | NVIDIA RTX 2060及以上（支持CUDA） | 无（CPU训练） |
| 存储 | 至少1GB可用空间 | 至少500MB可用空间 |

### 软件环境
| 项目 | 版本要求 |
|------|----------|
| 操作系统 | Windows 10/11 64位、Linux（Ubuntu 18.04+）、macOS 10.15+ |
| Python | 3.8 ~ 3.11 |
| TensorFlow | 2.8 ~ 2.15 |
| CUDA（可选） | 11.2 ~ 12.0（GPU加速时需要） |

### 依赖库版本
```bash
tensorflow>=2.8.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
```

## 安装依赖

### 方法一：使用pip直接安装
```bash
pip install tensorflow pandas numpy scikit-learn imbalanced-learn
```

### 方法二：使用requirements.txt（推荐）
创建 `requirements.txt` 文件：
```txt
tensorflow>=2.8.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
```

然后运行：
```bash
pip install -r requirements.txt
```

### GPU加速配置（可选）
如果使用NVIDIA GPU进行加速，需要安装CUDA和cuDNN：
1. 安装CUDA Toolkit：https://developer.nvidia.com/cuda-toolkit
2. 安装cuDNN：https://developer.nvidia.com/cudnn
3. 确保TensorFlow使用GPU：
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## 使用方法

### 数据准备
数据文件已随仓库提供，位于项目根目录：
- gene_seq.csv
- mirna_seq.csv
- Train_test.csv

### 运行代码

```bash
python mirna_gene_hybrid.py
```

### 输出结果

程序会输出：
- 最佳分类阈值
- F1得分（主要评估指标）
- 加权F1得分
- AUC值
- 分类报告（精确率、召回率、F1）
- 混淆矩阵

## 项目结构

```
miRNA/
├── mirna_gene_hybrid.py    # 主程序文件（深度学习模型）
├── gene_seq.csv           # 基因序列数据
├── mirna_seq.csv          # miRNA序列数据
├── Train_test.csv         # 关系对训练/测试数据
├── requirements.txt       # 依赖库列表（可选）
└── README.md              # 项目说明文档
```

## 代码结构说明

### 主程序文件 `mirna_gene_hybrid.py`

| 函数名 | 功能说明 |
|--------|----------|
| `get_kmer_features()` | 提取序列的k-mer频率特征 |
| `get_gc_content()` | 计算序列的GC含量 |
| `prepare_features()` | 准备特征矩阵 |
| `build_deep_learning_model()` | 构建深度学习模型 |
| `main()` | 主函数，执行完整的训练和评估流程 |

### 数据处理流程

```
数据加载 → 特征提取 → 预处理 → 数据集划分 → SMOTE过采样 → 模型训练 → 阈值寻优 → 结果评估
```

## 评估指标

本项目使用**F1得分**作为主要评估指标：

\[ F1 = 2 \times \frac{precision \times recall}{precision + recall} \]

其中：
- **precision**（精确率）：预测为正例的样本中真正正例的比例
- **recall**（召回率）：真正正例中被正确预测的比例

## 结果示例

```
=== 模型评估结果 ===
最佳阈值: 0.587
测试集F1得分: 0.6552
测试集加权F1得分: 0.8098
测试集AUC: 0.7670

分类报告:
                    precision    recall  f1-score   support

    Functional MTI       0.83      0.94      0.88        77
Non-Functional MTI       0.79      0.56      0.66        34

混淆矩阵:
[[72  5]
 [15 19]]
```

## 注意事项

1. 首次运行需要安装依赖库
2. 确保数据文件路径正确
3. 训练过程需要一定时间，具体取决于硬件配置
4. 建议使用GPU加速训练以提高效率
5. 数据集较小（738对关系），可能存在一定的随机性

## 许可证

本项目仅供学习和研究使用。

## 仓库说明

- 数据集 `gene_seq.csv`、`mirna_seq.csv`、`Train_test.csv` 已随仓库提供，可直接运行。
- 架构示意图见 [`docs/workflow.png`](docs/workflow.png)。
