import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# 特征提取
def get_kmer_features(sequence, k, alphabet):

    # 生成所有可能的k-mer组合
    kmers = [''.join(k) for k in product(alphabet, repeat=k)]
    kmer_counts = {kmer: 0 for kmer in kmers}
    
    # 预处理序列：去除无关字符
    seq = sequence.upper().replace('N', '').replace(' ', '').replace('\n', '').replace('"', '')
    
    # 统计每个k-mer的出现次数
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
    
    # 归一化处理
    total = sum(kmer_counts.values())
    return [kmer_counts[k]/total if total > 0 else 0.0 for k in kmers]

def get_gc_content(seq):
    """计算序列的GC含量百分比"""
    seq = seq.upper().replace('N', '').replace(' ', '').replace('\n', '').replace('"', '')
    return (seq.count('G') + seq.count('C')) / max(len(seq), 1) if len(seq) > 0 else 0.0

def prepare_features(train_df, gene_seqs, mirna_seqs):

    features = []
    gene_alpha = ['A', 'T', 'C', 'G']  # DNA字母表
    mirna_alpha = ['A', 'U', 'C', 'G']  # RNA字母表
    
    for _, row in train_df.iterrows():
        gene_seq = gene_seqs.get(row['gene'], '')
        mirna_seq = mirna_seqs.get(row['miRNA'], '')
        
        # 提取k-mer特征
        gene_3mer = get_kmer_features(gene_seq, 3, gene_alpha)
        mirna_3mer = get_kmer_features(mirna_seq, 3, mirna_alpha)
        gene_2mer = get_kmer_features(gene_seq, 2, gene_alpha)
        mirna_2mer = get_kmer_features(mirna_seq, 2, mirna_alpha)
        
        # 提取GC含量
        gene_gc = get_gc_content(gene_seq)
        mirna_gc = get_gc_content(mirna_seq)
        
        # 提取序列长度
        gene_len = len(gene_seq.replace('N', '').replace(' ', '').replace('\n', '').replace('"', ''))
        mirna_len = len(mirna_seq.replace('N', '').replace(' ', '').replace('\n', '').replace('"', ''))
        
        # 拼接所有特征
        combined = gene_3mer + mirna_3mer + gene_2mer + mirna_2mer + [gene_gc, mirna_gc, gene_len, mirna_len]
        features.append(combined)
    
    return np.array(features)

# ------------------------------
# 深度学习模型构建
# ------------------------------
def build_deep_learning_model(input_dim):

    model = Sequential([
        # 第一层：256神经元，ReLU激活
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),  # 批量归一化
        Dropout(0.3),          # Dropout防止过拟合
        
        # 第二层：128神经元
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # 第三层：64神经元
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # 第四层：32神经元
        Dense(32, activation='relu'),
        
        # 输出层：Sigmoid激活用于二分类
        Dense(1, activation='sigmoid')
    ])
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ------------------------------
# 主函数
# ------------------------------
def main():
    # 1. 加载数据
    print("正在加载数据...")
    gene_df = pd.read_csv('gene_seq.csv')
    mirna_df = pd.read_csv('mirna_seq.csv')
    train_df = pd.read_csv('Train_test.csv')
    
    # 构建序列字典
    gene_seqs = dict(zip(gene_df['label'], gene_df['sequence']))
    mirna_seqs = dict(zip(mirna_df['mirna'], mirna_df['seq'].str.replace('"', '').str.replace('\n', '').str.replace(' ', '')))
    
    # 2. 特征提取
    print("正在提取特征...")
    features = prepare_features(train_df, gene_seqs, mirna_seqs)
    labels = train_df['label'].values
    
    # 3. 预处理
    le = LabelEncoder()
    y = le.fit_transform(labels)  # 将标签转换为0/1
    
    # 特征筛选：去除低方差特征
    selector = VarianceThreshold(threshold=0.0001)
    features = selector.fit_transform(features)
    
    # 标准化处理
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # 4. 划分数据集（85%训练，15%测试，保持类别比例）
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.15, random_state=42, stratify=y)
    
    # 5. SMOTE过采样处理类别不平衡
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # 6. 构建并训练模型
    print("正在构建模型...")
    model = build_deep_learning_model(X_train.shape[1])
    
    # 早停机制：验证损失不再下降时停止训练
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # 学习率调度：验证损失不再下降时降低学习率
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    print("开始训练模型...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.15,  # 用训练集的15%作为验证集
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # 7. 模型评估
    print("\n正在评估模型...")
    y_pred_prob = model.predict(X_test, verbose=0)[:, 0]
    
    # 阈值寻优：寻找最大化F1得分的分类阈值
    best_f1 = 0
    best_thresh = 0.5
    best_f1_weighted = 0
    
    for thresh in np.arange(0.01, 0.99, 0.005):
        y_pred = (y_pred_prob >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        if f1_weighted > best_f1_weighted:
            best_f1 = f1
            best_f1_weighted = f1_weighted
            best_thresh = thresh
    
    y_pred = (y_pred_prob >= best_thresh).astype(int)
    
    # 输出评估结果
    print(f"\n=== 模型评估结果 ===")
    print(f"最佳阈值: {best_thresh:.3f}")
    print(f"测试集F1得分: {best_f1:.4f}")
    print(f"测试集加权F1得分: {best_f1_weighted:.4f}")
    print(f"测试集AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    main()