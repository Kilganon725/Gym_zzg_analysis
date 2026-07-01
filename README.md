# 基于 Python 的健身房客流时序数据挖掘与变化特征分析

## 项目简介

本项目是一个 Python 数据分析类毕业设计项目，围绕健身房客流时序数据展开，完成从原始数据读取、数据清洗、特征工程、探索性分析、预测模型训练到 Streamlit 可视化系统展示的完整流程。

项目题目：

```text
基于 Python 的健身房客流时序数据挖掘与变化特征分析
```

项目主要目标包括：

1. 对健身房客流或会员打卡数据进行数据概览与清洗。
2. 构造时间特征、滞后特征、滚动统计特征和变化率特征。
3. 分析客流在小时、星期、月份、工作日和周末等维度下的变化规律。
4. 构建客流预测模型，并使用 MAE、RMSE、MAPE、R2 等指标评价模型效果。
5. 使用 Streamlit 搭建可视化系统，辅助毕业论文展示和答辩演示。

## 数据来源说明

本项目使用的原始数据文件位于：

```text
data/raw/健身房会员打卡.csv
```

数据为健身房客流或会员打卡类时序数据，主要字段包括：

- `number_people`：客流人数，作为预测目标字段。
- `date`：日期时间字段，用于时间排序和时序分析。
- `timestamp`：当天秒数，用于辅助还原完整时间点。
- `day_of_week`：星期编号。
- `is_weekend`：是否为周末。
- `is_holiday`：是否为节假日。
- `temperature`：温度字段，用于分析外部因素与客流之间的关系。
- `month`、`hour`：月份和小时字段。

如果替换为其他健身房客流数据，建议 CSV 中至少包含时间字段和客流人数字段，例如 `date`、`datetime`、`timestamp`、`number_people`、`people`、`count`、`attendance`、`visitors` 等。

## 目录结构说明

```text
Gym_zzg_analysis/
  app/
    streamlit_app.py                  # Streamlit 可视化系统
  data/
    raw/
      健身房会员打卡.csv              # 原始数据
    processed/
      gym_flow_cleaned.csv            # 清洗后数据
      gym_flow_features.csv           # 特征工程后数据
      best_model.joblib               # 训练得到的最佳模型
  reports/
    figures/
      01_flow_time_series.png         # 客流时间序列图
      02_hourly_average_flow.png      # 每小时平均客流图
      03_weekday_average_flow.png     # 星期平均客流图
      04_workday_weekend_hourly_flow.png
      05_weekday_hour_heatmap.png
      06_flow_boxplot.png
      07_temperature_flow_scatter.png
      prediction_compare.png          # 真实值与预测值对比图
      feature_importance.png          # 特征重要性图
    tables/
      data_overview.txt               # 数据概览报告
      preprocess_log.txt              # 数据清洗日志
      eda_summary.txt                 # 探索性分析总结
      model_metrics.csv               # 模型评价指标
  src/
    data_overview.py                  # 原始数据概览
    data_preprocess.py                # 数据清洗
    feature_engineering.py            # 特征工程
    eda_analysis.py                   # 探索性分析与图表生成
    model_train.py                    # 模型训练与评估
  requirements.txt                    # Python 依赖
  README.md                           # 项目说明文档
```

## 环境安装命令

建议使用 Python 3.10 或以上版本。

在项目根目录下执行：

```bash
cd /Users/relphchris/Desktop/CodeX/zzg/Gym_zzg_analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如不使用虚拟环境，也可以直接执行：

```bash
pip install -r requirements.txt
```

## 数据处理运行命令

### 1. 原始数据概览

用于读取 `data/raw/` 下的第一个 CSV 文件，输出数据行数、列数、字段类型、缺失值、重复行和时间字段识别结果。

```bash
python3 src/data_overview.py
```

输出文件：

```text
reports/tables/data_overview.txt
```

### 2. 数据清洗

用于完成时间字段转换、重复值删除、缺失值填充、客流字段识别、负数异常处理和时间排序。

```bash
python3 src/data_preprocess.py
```

输出文件：

```text
data/processed/gym_flow_cleaned.csv
reports/tables/preprocess_log.txt
```

## 特征工程运行命令

用于构造小时、星期、月份、是否周末、滞后客流、滚动均值和变化率等特征。

```bash
python3 src/feature_engineering.py
```

输出文件：

```text
data/processed/gym_flow_features.csv
```

主要新增特征包括：

- `hour`
- `day_of_week`
- `day_name`
- `month`
- `is_weekend`
- `lag_1`
- `lag_2`
- `lag_3`
- `rolling_mean_3`
- `rolling_mean_6`
- `rolling_mean_24`
- `change_rate`

## 图表分析运行命令

用于生成探索性分析图表，并输出文字版分析总结。

```bash
python3 src/eda_analysis.py
```

输出目录：

```text
reports/figures/
reports/tables/eda_summary.txt
```

生成的主要图表包括：

1. `01_flow_time_series.png`：客流时间序列折线图。
2. `02_hourly_average_flow.png`：每小时平均客流柱状图。
3. `03_weekday_average_flow.png`：星期平均客流柱状图。
4. `04_workday_weekend_hourly_flow.png`：工作日与周末每小时客流对比折线图。
5. `05_weekday_hour_heatmap.png`：星期-小时客流热力图。
6. `06_flow_boxplot.png`：客流箱线图，用于观察异常值。
7. `07_temperature_flow_scatter.png`：温度与客流散点图。

## 模型训练运行命令

用于按时间顺序划分训练集和测试集，构建历史均值基准模型、随机森林回归模型和梯度提升回归模型，并输出模型评价结果。

```bash
python3 src/model_train.py
```

输出文件：

```text
reports/tables/model_metrics.csv
reports/figures/prediction_compare.png
reports/figures/feature_importance.png
data/processed/best_model.joblib
```

模型评价指标包括：

- `MAE`：平均绝对误差。
- `RMSE`：均方根误差。
- `MAPE`：平均绝对百分比误差。
- `R2`：决定系数。

当前结果中，表现最好的模型为 `GradientBoostingRegressor`，主要评价结果如下：

```text
MAE  = 3.0657
RMSE = 4.5012
MAPE = 32.5428
R2   = 0.9063
```

## Streamlit 系统启动命令

用于启动健身房客流分析可视化系统。

```bash
streamlit run app/streamlit_app.py
```

启动后浏览器访问：

```text
http://localhost:8501
```

系统包含以下页面：

1. 数据概览：展示数据量、字段数、时间范围和前几行数据。
2. 客流趋势：展示客流时间序列，并支持按小时、星期、月份查看平均客流。
3. 客流特征：展示工作日/周末对比图、星期-小时热力图、箱线图和温度散点图。
4. 预测结果：展示模型评价表、真实值与预测值对比图、特征重要性图。
5. 运营建议：根据分析结果自动生成高峰时段、低谷时段和管理建议。

## 论文中可以使用的主要图表和结果文件

### 第 3 章：数据来源与预处理

可使用文件：

- `reports/tables/data_overview.txt`
- `reports/tables/preprocess_log.txt`
- `data/processed/gym_flow_cleaned.csv`

可说明内容：

- 原始数据规模、字段类型和前 5 行样例。
- 缺失值、重复值和异常值处理情况。
- 清洗前后数据量变化。

### 第 4 章：客流变化特征分析

可使用文件：

- `reports/figures/01_flow_time_series.png`
- `reports/figures/02_hourly_average_flow.png`
- `reports/figures/03_weekday_average_flow.png`
- `reports/figures/04_workday_weekend_hourly_flow.png`
- `reports/figures/05_weekday_hour_heatmap.png`
- `reports/figures/06_flow_boxplot.png`
- `reports/figures/07_temperature_flow_scatter.png`
- `reports/tables/eda_summary.txt`

可说明内容：

- 健身房客流存在明显的时间波动和周期性。
- 不同小时、不同星期的平均客流存在差异。
- 工作日与周末客流规律不同。
- 温度等外部因素与客流之间存在一定关系。

### 第 5 章：客流预测模型

可使用文件：

- `reports/tables/model_metrics.csv`
- `reports/figures/prediction_compare.png`
- `reports/figures/feature_importance.png`
- `data/processed/best_model.joblib`

可说明内容：

- 历史均值基准模型、随机森林模型和梯度提升模型的对比结果。
- 最佳模型在测试集上的 MAE、RMSE、MAPE 和 R2。
- 真实客流与预测客流的拟合效果。
- 影响预测结果的重要特征。

### 第 6 章：可视化系统设计与实现

可使用文件：

- `app/streamlit_app.py`
- Streamlit 系统运行截图。

可说明内容：

- 系统页面结构。
- 数据概览、趋势分析、特征分析、预测结果和运营建议模块。
- 系统对毕业设计答辩展示和运营辅助决策的支持作用。

## 推荐完整运行顺序

从原始数据到可视化系统，推荐按以下顺序运行：

```bash
python3 src/data_overview.py
python3 src/data_preprocess.py
python3 src/feature_engineering.py
python3 src/eda_analysis.py
python3 src/model_train.py
streamlit run app/streamlit_app.py
```

如果已存在 `data/processed/gym_flow_features.csv`、`reports/figures/` 和 `reports/tables/model_metrics.csv`，可以直接启动 Streamlit 系统进行展示：

```bash
streamlit run app/streamlit_app.py
```
