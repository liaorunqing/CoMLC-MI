# CoMLC-MI 系统架构与数据集特征文档

> 完成于 2026-06-21，涵盖 126 个实验的完整研究历程

---

## 第一部分：数据集特征

### 1.1 基本信息

| 属性 | 值 |
|------|-----|
| 来源 | Krasnoyarsk Interdistrict Clinical Hospital No. 20, Russia |
| 时间窗口 | 1992–1995 |
| 患者总数 | 1,700 |
| 输入特征数 | 111（排除 CPK 和 IBS_NASL 后） |
| 输出标签数 | 12 个二分类并发症 |
| 总体缺失率 | 8.47% |
| 完全案例数 | 0（所有患者至少有一项特征缺失） |
| 平均标签基数 | 1.271（每名患者平均 1.27 个并发症） |

### 1.2 特征分组（111 个输入特征）

#### 人口学特征（2 个）
`AGE`（年龄）, `SEX`（性别）

#### 病史与共病（~18 个）
`INF_ANAM`（既往 MI 次数）, `STENOK_AN`（心绞痛功能分级）, `FK_STENOK`（心绞痛功能等级）, `IBS_POST`（既往 CAD 诊断）, `GB`（高血压分级）, `SIM_GIPERT`（症状性高血压）, `DLIT_AG`（动脉高血压持续时间）, `ZSN_A`（慢性心衰病史）,

伴随疾病标志物：`nr_01`–`nr_11`（多种伴随诊断）, `np_01`–`np_10`（既往并发症）, `endocr_01`–`endocr_03`（内分泌疾病：DM、肥胖、甲亢等）, `zab_leg_01`–`zab_leg_06`（外周动脉疾病）

#### 入院生理指标（~10 个）
`S_AD_KBRIG`（EMS 收缩压，63.3% 缺失，保留缺失指示变量）, `D_AD_KBRIG`（EMS 舒张压，63.3% 缺失）, `S_AD_ORIT`（入院收缩压，15.7% 缺失）, `D_AD_ORIT`（入院舒张压，15.7% 缺失）,

`O_L_POST`（入院肺水肿）, `K_SH_POST`（入院心源性休克）, `MP_TP_POST`（入院起搏器使用）, `SVT_POST` / `GT_POST` / `FIB_G_POST`（入院心律失常类型）

#### 心电图特征（~30 个）
MI 定位：`ant_im`, `lat_im`, `inf_im`, `post_im`（前壁/侧壁/下壁/后壁 MI），`IM_PG_P`（右室 MI）

心律特征：`ritm_ecg_p_01`–`ritm_ecg_p_08`（窦性、房颤、室上速、室速、室颤、AV 阻滞等）

ECG 形态：`n_r_ecg_p_01`–`n_r_ecg_p_10`（心率变异性等），`n_p_ecg_p_01`–`n_p_ecg_p_12`（ST 段变化、T 波异常、病理性 Q 波等）

#### 纤溶治疗（~7 个）
`fibr_ter_01`–`fibr_ter_08`（链激酶、tPA 等，给药时机和剂量）

#### 血液化验（9 个）
`GIPO_K`/`K_BLOOD`（低钾/血钾，21.8% 缺失）, `GIPER_NA`/`NA_BLOOD`（高钠/血钠，22.1% 缺失）, `ALT_BLOOD`/`AST_BLOOD`（肝酶，16.7% 缺失）, `L_BLOOD`（白细胞，7.4% 缺失）, `ROE`（血沉，11.9% 缺失）

#### 时序特征（~20 个，分三个 ICU 日）

**入院时间**：`TIME_B_S`（发病到入院时间，7.4% 缺失）

**ICU Day 1**：`R_AB_1_n`–`R_AB_3_n`（疼痛复发次数）, `NA_KB`/`NOT_NA_KB`（麻醉性镇痛药/非麻醉性镇痛药使用）, `LID_KB`（利多卡因）, `NITR_S`（硝酸酯类静脉给药）

**ICU Day 2**：`NA_R_1_n`–`NA_R_3_n`（第二天麻醉性镇痛药）, `NOT_NA_1_n`–`NOT_NA_3_n`（第二天非麻醉性镇痛药）

**ICU Day 3**：`LID_S_n`, `B_BLOK_S_n`, `ANT_CA_S_n`, `GEPAR_S_n`, `ASP_S_n`, `TIKL_S_n`, `TRENT_S_n`（口服药物：利多卡因、β受体阻滞剂、钙拮抗剂、肝素、阿司匹林、Ticlid、Trental）

### 1.3 输出标签（12 个并发症）

| 编码 | 英文名 | 中文名 | 患病率 | 病理分类 |
|------|--------|--------|--------|---------|
| FIBR_PREDS | Atrial Fibrillation | 房颤 | 10.0% | 心律失常 |
| PREDS_TAH | Supraventricular Tachycardia | 室上速 | 1.2% | 心律失常 |
| JELUD_TAH | Ventricular Tachycardia | 室速 | 2.5% | 心律失常 |
| FIBR_JELUD | Ventricular Fibrillation | 室颤 | 4.2% | 心律失常 |
| A_V_BLOK | 3rd-degree AV Block | 三度房室传导阻滞 | 3.4% | 心律失常 |
| OTEK_LANC | Pulmonary Edema | 肺水肿 | 9.4% | 血流动力学 |
| RAZRIV | Myocardial Rupture | 心肌破裂 | 3.2% | 结构性 |
| DRESSLER | Dressler Syndrome | 心肌梗死后综合征 | 4.4% | 免疫性 |
| ZSN | Chronic Heart Failure | 慢性心衰 | 23.2% | 血流动力学 |
| REC_IM | Recurrent MI | 再发心梗 | 9.4% | 复发 |
| P_IM_STEN | Post-infarction Angina | 梗死后心绞痛 | 8.7% | 复发 |
| LET_IS | Lethal Outcome | 致死结局 | 15.9% | 终点 |

> 注：LET_IS 原始数据为序数变量（0=存活, 1–7=不同类型的死亡），在预处理中二值化为 0/1（存活/死亡）。

### 1.4 标签共现结构

**Top-10 条件概率 P(Col | Row)：**

| 来源并发症 | 目标并发症 | P |
|-----------|-----------|-----|
| Myocardial Rupture | Lethal Outcome | 1.000 |
| Pulmonary Edema | Chronic Heart Failure | 0.410 |
| Recurrent MI | Chronic Heart Failure | 0.340 |
| Atrial Fibrillation | Chronic Heart Failure | 0.308 |
| Dressler Syndrome | Chronic Heart Failure | 0.298 |
| Ventricular Fibrillation | Lethal Outcome | 0.288 |
| Recurrent MI | Lethal Outcome | 0.283 |
| 3rd-degree AV Block | Lethal Outcome | 0.273 |
| Ventricular Tachycardia | Chronic Heart Failure | 0.259 |
| Pulmonary Edema | Recurrent MI | 0.238 |

**关键统计**：
- 中位条件概率 < 0.15
- 18 对共现关系超过 P=0.2
- CHF 和 Lethal Outcome 是最主要的"汇聚节点"
- 该共现结构过于稀疏，不足以支撑图神经网络的消息传递机制

---

## 第二部分：系统架构与构建思路

### 2.1 总体研究设计

本研究旨在建立心肌梗死并发症多标签预测的系统基准。研究经历了四个迭代阶段：

```
v0（原始）      基线建立 + GCN 消融
  ↓
v1（改进尝试）  MissForest + ECC + MLSMOTE
  ↓
v2（代码修复）  5 个 Bug 修正 + CatBoost + 鲁棒阈值
  ↓
v3（深度优化）  集成方法 + 临床特征 + 患者图 GAT（无效）
  ↓
最终修正         验证集真实权重 + DeLong/排列检验 + Bootstrap
```

### 2.2 数据预处理流水线

```
原始 CSV (1700 × 124)
  │
  ├─ 排除特征：CPK (99.8%缺失), IBS_NASL (95.8%缺失)
  ├─ LET_IS 二值化：序数 0-7 → 二分类 0/1
  ├─ EMS 血压缺失指示变量：S_AD_KBRIG_MISSING, D_AD_KBRIG_MISSING
  │
  ├─ 分层划分：70% 训练 / 10% 验证 / 20% 测试（按 LET_IS 分层）
  │
  ├─ MICE 插补（13 个 10-65% 缺失特征，RF 估算器，10 次迭代）
  │   仅对训练集 fit，验证集和测试集 transform（无数据泄露）
  │
  ├─ 中位数/众数插补（其余 <10% 缺失特征）
  │
  └─ StandardScaler 标准化（仅对连续数值特征，二元 0/1 特征不缩放）
      关键：单个 Scaler 实例，fit 仅在训练集，transform 验证集和测试集
```

### 2.3 核心模块架构

```
src/
├── config.py                    ← 全局配置中心
│   ├── 所有路径、标签定义、特征列表
│   └── 超参数默认值、TabPFN 设置
│
├── evaluation.py                ← 统一评估框架
│   ├── compute_all_metrics()    → Micro/Macro-AUC, AUPRC, F1, Hamming Loss
│   ├── bootstrap CI             → 百分位 Bootstrap 500 次重采样
│   ├── robust_threshold_tuning()→ 5 折 CV 内中位数最优阈值
│   └── print_metrics_table()   → 格式化输出
│
├── preprocessing.py             ← 数据预处理
│   └── MICE + 缺失指示变量 + Scaler
│
└── [实验脚本]
    ├── baselines.py             → 8 个基线模型
    ├── comlc_mi_opt.py          → 18 组 GCN+CAL 消融
    ├── temporal_experiments.py  → 4 个时间窗口实验
    ├── shap_analysis.py         → SHAP 可解释性
    ├── improvements_v2.py       → Bug 修复后的 6 模型对比
    ├── publication_fixes.py     → 2 模型集成 + Bootstrap + Stacking
    └── final_statistical_tests.py → DeLong + 排列检验 + FDR
```

### 2.4 模型体系（阶段性演进）

#### 基线模型（v0，8 个）
- **独立分类器**：XGBoost（每标签独立训练，scale_pos_weight 处理不平衡）
- **经典多标签方法**：Binary Relevance (LR), Classifier Chains (3 链集成), Label Powerset (RF), RAkEL (k=4)
- **深度多标签**：Shared-Encoder MLP (256→128→64, 加权 BCE), Multitask DNN（共享编码器 + 每标签分支）

#### CoMLC-MI 架构探索（v0，已证实无效）
- **特征编码器**：MLP (111→256→128→64)，BatchNorm + ReLU + Dropout
- **标签共现 GCN**：2 层 GCNConv 在 12 节点标签图上传播
- **预测头**：患者嵌入 h 与标签嵌入 z̃ 的点积 h·z̃
- **临床非对称损失 (CAL)**：γ⁺=0, γ⁻=3，正样本权重 1/prevalence(k)
- **消融维度**：图结构 (none/sparse/symmetric) × 架构 (dot/concat) × 损失 (W_BCE/CAL_g2/CAL_g3) = 18 配置
- **结论**：GCN 在所有 18 配置中均降低性能，CAL 在所有配置中均劣于加权 BCE

#### 时序分析（v0）
- 4 个预测窗口：入院 (t0, 91 特征)、24h (t1, 98 特征)、48h (t2, 104 特征)、72h (t3, 111 特征)
- 发现 24h 为最优预测窗口，ICU Day 1 数据贡献最大边际增益

#### 改进后方法（v2–v3）

**LP-RF（Label Powerset + 随机森林）**
```
输入特征 → RF (200 trees, max_depth=10)
         → 多标签输出 (12 维概率向量)
增强：true MLSMOTE（多标签插值，特征+标签联合空间）
```

**ECC-LGBM（集成分类器链）**
```
50 条并行链，每条链：
  随机标签顺序 → 链式 ClassifierChain
  → LightGBM (200 trees, max_depth=6, lr=0.03)
  → 前一个标签的预测作为下一个标签的附加特征
50 条链的输出取均值
```

**TabPFN v2-BR（表格基础模型 + 二元相关性）**
```
每标签独立 TabPFN：
  top-80 特征（12 标签 RF 重要性聚合）
  → TabPFNClassifier (零超参调优, 全量 1360 训练样本)
  → 12 次独立预测 → 拼接为 (N, 12) 概率矩阵
```

**CatBoost-BR（声明类别特征 + 有序提升）**
```
每标签独立 CatBoost：
  80 个二元 (0/1) 特征声明为 cat_features
  → class_weights=[1.0, n_neg/n_pos]
  → ordered boosting + Iter 型 early stopping (od_wait=50)
```

**两模型集成（最终 SOTA）**
```
LP-RF 预测概率 (擅长极稀有心律失常：SVT, VT, AVB)
  +
TabPFN 预测概率 (擅长中频并发症：AF, PulEd, Leth)
  →
  等权平均 (1/2, 1/2)
  → Macro-AUC = 0.7604, Micro-AUC = 0.8140
```

### 2.5 统计推断体系

```
每标签 DeLong 检验（n+ ≥ 15 的标签）
  ├── H₀: AUC(model_A) = AUC(model_B)
  ├── DeLong 协方差矩阵（患者配对结构）
  └── 精确 p 值（不截断为 <0.0001）

每标签排列检验（n+ < 15 的标签）
  ├── 10,000 次随机交换预测值（配对排列）
  ├── 无渐近假设，适用于任意样本量
  └── p = (极端排列数 + 1) / (排列总数 + 1)

Benjamini-Hochberg FDR 校正
  ├── 12 个标签的 p 值合并 → p_fdr = p_raw × 12 / rank
  └── α = 0.05

Bootstrap 稳健性
  ├── 500 次有放回重采样测试集
  └── 报告均值 + 95% 百分位 CI
```

### 2.6 Bug 修复历程（5 个关键修复）

| Bug | 严重度 | 影响 | 修复后提升 |
|-----|--------|------|-----------|
| Scaler 数据泄露（3 个独立 StandardScaler） | 🔴 | 验证集/测试集使用自身统计量归一化 | LP-RF +3.07 pts |
| TabPFN 训练样本截断至 1000 | 🔴 | 丢失 26% 训练数据 | TabPFN +3.27 pts |
| 假 MLSMOTE（单标签 imblearn.SMOTE） | 🔴 | 多标签过采样退化为单标签 | SVT AUC +31.32 pts |
| ECC 串行训练 + 弱超参 | 🟡 | 50 链串行执行，n_estimators 不足 | 20× 加速 |
| 集成权重使用硬编码占位符 | 🔴 | 论文核心 SOTA 数字基于虚假权重 | 全部重新测量 |

### 2.7 关键发现总结

**生效的方法：**
1. 两模型等权集成 (LP-RF + TabPFN)：Macro-AUC=0.7604，超越最佳单模型
2. 真实 MLSMOTE 针对稀有心律失常标签
3. CatBoost + 类别特征声明：最高 Macro-F1
4. 5 折 CV 鲁棒阈值调优（罕见标签 t* 可低至 0.001）
5. Bootstrap 稳健性检验缓解单数据集质疑

**证实无效的方法：**
1. 标签共现 GCN（18 组消融全部劣于基线）
2. 患者相似性 GAT（9 组图参数扫描，最佳 val=0.7223）
3. 临床非对称损失 CAL（γ⁻ 聚焦 + 1/w 逆权重双重惩罚）
4. AUC 加权集成 vs 等权集成（差异仅 0.0001，Softmax 区分力不足）

**被证伪的结论（经排列检验纠正）：**
1. "TabPFN 在 SVT 上显著优于集成" → DeLong 假阳性（n+=4），排列检验 p=0.774
2. "TabPFN 在 AVB 上显著优于集成" → DeLong 假阳性（n+=9），排列检验 p=0.977

**稳健的临床发现：**
1. CHF 的 4/4 SHAP 特征验证（AGE, ZSN_A, FK_STENOK, DLIT_AG）
2. ICU Day 1 数据为最优时序预测窗口
3. 心律失常在入院时即可预测，血流动力学并发症随 ICU 数据积累改善

---

## 第三部分：复现指南

### 3.1 环境要求
- Python 3.12, NumPy 1.26, scikit-learn 1.7, LightGBM 4.6, CatBoost, TabPFN, PyTorch Geometric, SHAP

### 3.2 最小复现步骤

```bash
# 1. 数据预处理
python src/preprocessing.py
# 输出: output/processed_data/{X,y}_{train,val,test}.csv

# 2. 训练所有模型
python src/publication_fixes.py
# 输出: output/improvements_v2/PUB_*_preds.npy

# 3. 统计检验
python src/final_statistical_tests.py
# 输出: output/improvements_v2/FINAL_STATISTICAL_RESULTS.json
```

### 3.3 最终数字（可引用）

| 模型 | Micro-AUC | Macro-AUC | Macro-AUPRC | 权重来源 |
|------|-----------|-----------|-------------|---------|
| Ensemble 2-model (equal) | **0.8140** | **0.7604** | 0.2642 | 等权 |
| TabPFN v2-BR | 0.8099 | 0.7587 | 0.2564 | 单模型最优 |
| LP-RF | 0.8050 | 0.7570 | 0.2516 | Macro-AUC 单模型第二 |
| ECC-LGBM | 0.7884 | 0.6960 | 0.2240 | — |
| Ensemble 3-model | 0.8123 | 0.7514 | 0.2559 | 三模型集成（已弃用） |

> 所有数字基于 340 人独立测试集，集成权重基于 170 人验证集实测 AUC，经 Bootstrap 500 次稳健性检验，每标签显著性经排列检验（n+<15）或 DeLong 检验（n+≥15）+ BH-FDR 校正。

---

*文档完成于 2026-06-21，涵盖全部 126 个实验。*
