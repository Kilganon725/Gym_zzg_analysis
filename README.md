# 基于 Python 的健身房客流时序数据挖掘与变化特征分析

本项目是一个 Python 数据分析类毕业设计项目，围绕健身房会员打卡或客流时序数据展开。项目目标是通过数据清洗、特征工程、探索性分析、客流预测模型和可视化系统，分析健身房客流在不同时间维度下的变化规律，并为健身房运营管理提供参考。

## 项目主题

题目：基于 Python 的健身房客流时序数据挖掘与变化特征分析

主要研究内容：

1. 对健身房客流或会员打卡数据进行清洗和预处理。
2. 构造小时、星期、月份、是否周末、滞后客流、滚动均值等时序特征。
3. 分析健身房客流的日内变化、周内变化、工作日与周末差异等规律。
4. 构建客流预测模型，并使用 MAE、RMSE、MAPE、R2 等指标评价效果。
5. 使用 Streamlit 实现可视化展示系统，辅助毕业设计答辩展示。

## 目录结构

```text
zzg/
  data/
    raw/                 # 原始数据文件
    processed/           # 清洗后数据、特征数据、模型文件
  notebooks/             # Jupyter Notebook 探索分析文件
  src/                   # 数据处理、特征工程、建模等 Python 脚本
  app/                   # Streamlit 可视化系统
  reports/
    figures/             # 图表输出
    tables/              # 表格、日志、模型评价结果输出
  README.md              # 项目说明
  requirements.txt       # Python 依赖
```

## 环境准备

建议使用 Python 3.10 或以上版本。

在项目根目录下执行：

```bash
cd /Users/relphchris/Desktop/CodeX/zzg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据准备

将原始 CSV 数据放入：

```text
data/raw/
```

推荐数据文件类型：

- 会员打卡记录数据
- 健身房客流人数数据
- 带有时间字段的健身房签到或出勤数据

数据中最好包含：

- 时间字段，例如 `timestamp`、`datetime`、`date`、`checkin_time`
- 客流或签到人数字段，例如 `number_people`、`count`、`attendance`、`visitors`
- 可选外部字段，例如温度、天气、会员类型、课程类型等

## 运行步骤

后续脚本开发完成后，建议按以下顺序运行：

```bash
python src/data_overview.py
python src/data_preprocess.py
python src/feature_engineering.py
python src/eda_analysis.py
python src/model_train.py
streamlit run app/streamlit_app.py
```

各步骤作用：

1. `data_overview.py`：读取原始数据，输出字段、缺失值、重复值等数据概览。
2. `data_preprocess.py`：完成数据清洗，生成清洗后的数据文件。
3. `feature_engineering.py`：构造时间特征、滞后特征和滚动统计特征。
4. `eda_analysis.py`：生成客流趋势、小时规律、星期规律、热力图等分析图表。
5. `model_train.py`：训练客流预测模型并输出评价指标。
6. `streamlit_app.py`：启动可视化系统，用于展示分析结果和预测结果。

## 预期输出

项目运行后主要输出：

- `data/processed/gym_flow_cleaned.csv`：清洗后数据
- `data/processed/gym_flow_features.csv`：特征工程后数据
- `data/processed/best_model.joblib`：最佳预测模型
- `reports/figures/`：论文和系统展示使用的图表
- `reports/tables/`：数据概览、清洗日志、模型评价结果等
- Streamlit 可视化系统页面

## 后续开发计划

### 第一阶段：数据理解与清洗

- 整理原始数据字段。
- 识别时间字段和客流字段。
- 处理缺失值、重复值和异常值。
- 输出数据概览和清洗日志。

### 第二阶段：特征工程

- 提取小时、星期、月份、是否周末等时间特征。
- 构造滞后客流特征。
- 构造滚动均值和客流变化率特征。

### 第三阶段：探索性分析

- 绘制客流时间序列图。
- 分析每小时平均客流。
- 分析星期维度客流变化。
- 对比工作日与周末客流差异。
- 绘制星期和小时交叉热力图。

### 第四阶段：预测模型

- 构建历史均值基准模型。
- 构建随机森林回归模型。
- 构建梯度提升回归模型。
- 对比不同模型的 MAE、RMSE、MAPE、R2。
- 保存最佳模型和预测结果图。

### 第五阶段：可视化系统

- 使用 Streamlit 搭建系统页面。
- 实现数据概览、趋势分析、特征分析、预测结果和运营建议模块。
- 将系统截图和图表用于毕业论文第 6 章。

## 论文对应关系

本项目代码和结果可支撑毕业论文：

- 第 3 章：数据来源与预处理
- 第 4 章：健身房客流变化特征分析
- 第 5 章：健身房客流预测模型
- 第 6 章：可视化系统设计与实现

