"""
健身房客流预测模型训练与评估脚本。

读取 data/processed/gym_flow_features.csv，按照时间顺序划分训练集和测试集，
训练历史均值基准模型、随机森林和梯度提升回归模型，输出模型评价结果、
预测对比图、特征重要性图，并保存最佳模型。
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "gym_flow_features.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "best_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "tables" / "model_metrics.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
PREDICTION_COMPARE_PATH = FIGURE_DIR / "prediction_compare.png"
FEATURE_IMPORTANCE_PATH = FIGURE_DIR / "feature_importance.png"


class HistoricalMeanModel:
    """历史均值基准模型，用训练集目标均值作为所有测试样本的预测值。"""

    def __init__(self) -> None:
        self.mean_value: float | None = None

    def fit(self, y_train: pd.Series) -> "HistoricalMeanModel":
        self.mean_value = float(y_train.mean())
        return self

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        if self.mean_value is None:
            raise ValueError("历史均值模型尚未训练。")
        return np.full(shape=len(x_test), fill_value=self.mean_value)


def set_chinese_font() -> None:
    """设置常见中文字体，尽量避免图表中文乱码。"""
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font=plt.rcParams["font.sans-serif"][0])


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    """使用常见编码读取 CSV，提高中文文件和带 BOM 文件的兼容性。"""
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"CSV 编码识别失败，已尝试：{', '.join(encodings)}",
    )


def identify_flow_column(df: pd.DataFrame) -> str:
    """根据字段名自动识别客流人数字段作为预测目标。"""
    flow_keywords = (
        "number_people",
        "people",
        "count",
        "attendance",
        "visitors",
        "visitor",
        "flow",
        "traffic",
        "人数",
        "客流",
        "人流",
        "到店",
        "打卡",
    )

    for column in df.columns:
        column_name = str(column).lower()
        if any(keyword in column_name for keyword in flow_keywords):
            if is_numeric_dtype(df[column]):
                return column

    raise ValueError(
        "未能识别客流人数字段。请确认数据中包含 number_people、people、count、"
        "attendance、visitors、客流、人数等数值字段。"
    )


def prepare_analysis_datetime(df: pd.DataFrame) -> str:
    """
    构造排序用时间字段。

    特征文件中的 date 可能只有日期粒度；如果存在 0-86399 范围内的 timestamp，
    则组合为“日期 + 当天秒数”，用于严格按时间顺序划分训练/测试集。
    """
    if "date" not in df.columns:
        raise ValueError("未找到 date 字段，无法按时间顺序划分训练集和测试集。")

    base_time = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    if base_time.notna().mean() < 0.8:
        raise ValueError("date 字段无法可靠转换为 datetime 类型。")

    analysis_column = "analysis_datetime"
    has_date_only = (
        base_time.dt.hour.eq(0)
        & base_time.dt.minute.eq(0)
        & base_time.dt.second.eq(0)
    ).all()

    if has_date_only and "timestamp" in df.columns and is_numeric_dtype(df["timestamp"]):
        timestamp_values = pd.to_numeric(df["timestamp"], errors="coerce")
        if timestamp_values.between(0, 86_399).mean() >= 0.8:
            df[analysis_column] = base_time.dt.normalize() + pd.to_timedelta(timestamp_values, unit="s")
        else:
            df[analysis_column] = base_time
    else:
        df[analysis_column] = base_time

    return analysis_column


def select_feature_columns(df: pd.DataFrame, target_column: str, time_column: str) -> list[str]:
    """
    选择建模特征列。

    为避免数据泄露，排除预测目标、日期文本、绘图时间列，以及由当前目标值计算的
    rolling_mean_* 和 change_rate；保留 lag_* 等只依赖历史客流的特征。
    """
    leakage_or_non_feature_columns = {
        target_column,
        "date",
        "day_name",
        "day_name_cn",
        time_column,
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_mean_24",
        "change_rate",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in leakage_or_non_feature_columns and is_numeric_dtype(df[column])
    ]

    if not feature_columns:
        raise ValueError("未找到可用于模型训练的数值特征列。")

    return feature_columns


def split_by_time(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    time_column: str,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """按时间顺序划分训练集和测试集，前 80% 为训练集，后 20% 为测试集。"""
    df = df.sort_values(time_column).reset_index(drop=True)
    split_index = int(len(df) * train_ratio)

    if split_index <= 0 or split_index >= len(df):
        raise ValueError("数据量不足，无法按 80%/20% 划分训练集和测试集。")

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    x_train = train_df[feature_columns]
    x_test = test_df[feature_columns]
    y_train = train_df[target_column]
    y_test = test_df[target_column]
    test_time = test_df[time_column]

    return x_train, x_test, y_train, y_test, test_time


def mean_absolute_percentage_error(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """计算 MAPE，真实值为 0 的样本不参与百分比误差计算。"""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    non_zero_mask = y_true_array != 0

    if not non_zero_mask.any():
        return np.nan

    return float(
        np.mean(
            np.abs((y_true_array[non_zero_mask] - y_pred_array[non_zero_mask]) / y_true_array[non_zero_mask])
        )
        * 100
    )


def evaluate_model(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | str]:
    """使用 MAE、RMSE、MAPE、R2 评价模型。"""
    return {
        "model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def train_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """训练历史均值、随机森林和梯度提升模型。"""
    models = {
        "HistoricalMean": HistoricalMeanModel().fit(y_train),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    predictions = {}
    for model_name, model in models.items():
        if model_name != "HistoricalMean":
            model.fit(x_train, y_train)
        predictions[model_name] = model.predict(x_test)

    return models, predictions


def plot_prediction_compare(
    test_time: pd.Series,
    y_test: pd.Series,
    predictions: dict[str, np.ndarray],
    best_model_name: str,
) -> None:
    """绘制测试集真实值与最佳模型预测值对比图。"""
    plot_df = pd.DataFrame(
        {
            "time": test_time,
            "真实值": y_test.to_numpy(),
            "预测值": predictions[best_model_name],
        }
    ).sort_values("time")

    # 测试集较长时抽样展示，保证论文图表清晰可读。
    if len(plot_df) > 1000:
        plot_df = plot_df.iloc[-1000:]

    plt.figure(figsize=(13, 5))
    plt.plot(plot_df["time"], plot_df["真实值"], label="真实值", linewidth=1.0, color="#2563eb")
    plt.plot(plot_df["time"], plot_df["预测值"], label="预测值", linewidth=1.0, color="#dc2626", alpha=0.9)
    plt.title(f"真实客流与预测客流对比（{best_model_name}）")
    plt.xlabel("时间")
    plt.ylabel("客流人数")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(PREDICTION_COMPARE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(best_model: object, feature_columns: list[str]) -> None:
    """绘制最佳树模型的特征重要性图。"""
    if not hasattr(best_model, "feature_importances_"):
        return

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": best_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df = importance_df.head(20)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", color="#16a34a")
    plt.title("特征重要性")
    plt.xlabel("重要性")
    plt.ylabel("特征")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    """主流程：读取数据、训练模型、评估并保存结果。"""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"未找到输入文件：{INPUT_PATH}。请先运行 python3 src/feature_engineering.py。"
        )

    set_chinese_font()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = read_csv_with_fallback(INPUT_PATH)
    target_column = identify_flow_column(df)
    time_column = prepare_analysis_datetime(df)
    df = df.dropna(subset=[target_column, time_column]).copy()

    feature_columns = select_feature_columns(df, target_column, time_column)
    x_train, x_test, y_train, y_test, test_time = split_by_time(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        time_column=time_column,
    )

    models, predictions = train_models(x_train, x_test, y_train)

    metrics = [
        evaluate_model(model_name, y_test, y_pred)
        for model_name, y_pred in predictions.items()
    ]
    metrics_df = pd.DataFrame(metrics).sort_values("RMSE").reset_index(drop=True)
    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")

    best_model_name = str(metrics_df.iloc[0]["model"])
    best_model = models[best_model_name]

    joblib.dump(
        {
            "model_name": best_model_name,
            "model": best_model,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "time_column": time_column,
            "metrics": metrics_df.to_dict(orient="records"),
            "note": "训练特征已排除包含当前目标值的 rolling_mean_* 和 change_rate，以避免数据泄露。",
        },
        MODEL_PATH,
    )

    plot_prediction_compare(test_time, y_test, predictions, best_model_name)
    plot_feature_importance(best_model, feature_columns)

    print("健身房客流预测模型训练完成")
    print("=" * 40)
    print(f"输入文件：{INPUT_PATH}")
    print(f"预测目标字段：{target_column}")
    print(f"训练集样本数：{len(x_train)}")
    print(f"测试集样本数：{len(x_test)}")
    print(f"建模特征：{', '.join(feature_columns)}")
    print("\n模型评价结果：")
    print(metrics_df.to_string(index=False))
    print(f"\n最佳模型：{best_model_name}")
    print(f"评价结果已保存至：{METRICS_PATH}")
    print(f"预测对比图已保存至：{PREDICTION_COMPARE_PATH}")
    if hasattr(best_model, "feature_importances_"):
        print(f"特征重要性图已保存至：{FEATURE_IMPORTANCE_PATH}")
    else:
        print("最佳模型不支持 feature_importances_，未生成特征重要性图。")
    print(f"最佳模型已保存至：{MODEL_PATH}")


if __name__ == "__main__":
    main()
