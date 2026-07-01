"""
健身房客流探索性分析脚本。

读取 data/processed/gym_flow_features.csv，生成客流时序、小时规律、
星期规律、工作日/周末对比、星期-小时热力图、箱线图和温度散点图，
并将图表保存到 reports/figures/，分析结论保存到 reports/tables/eda_summary.txt。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "gym_flow_features.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
SUMMARY_PATH = TABLE_DIR / "eda_summary.txt"

DAY_NAME_MAP = {
    0: "周一",
    1: "周二",
    2: "周三",
    3: "周四",
    4: "周五",
    5: "周六",
    6: "周日",
}


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
    """根据字段名自动识别客流人数字段。"""
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


def identify_datetime_column(df: pd.DataFrame) -> str:
    """识别可用于时间序列分析的时间字段。"""
    time_keywords = ("datetime", "date", "time", "timestamp", "日期", "时间")

    for column in df.columns:
        column_name = str(column).lower()
        if not any(keyword in column_name for keyword in time_keywords):
            continue
        if is_numeric_dtype(df[column]):
            continue

        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            df[column] = parsed
            return column

    raise ValueError(
        "未能识别时间字段。请确认数据中包含 date、datetime、time、日期或时间等可解析字段。"
    )


def prepare_analysis_time(df: pd.DataFrame, datetime_column: str) -> str:
    """
    构造绘图用完整时间轴。

    若 date 只有日期且存在 0-86399 范围内的 timestamp，则组合为日期+当天秒数；
    否则直接使用识别到的时间字段。
    """
    analysis_column = "analysis_datetime"
    base_time = pd.to_datetime(df[datetime_column], errors="coerce").dt.tz_localize(None)

    has_date_only = (base_time.dt.hour.eq(0) & base_time.dt.minute.eq(0) & base_time.dt.second.eq(0)).all()
    if has_date_only and "timestamp" in df.columns and is_numeric_dtype(df["timestamp"]):
        timestamp_values = pd.to_numeric(df["timestamp"], errors="coerce")
        if timestamp_values.between(0, 86_399).mean() >= 0.8:
            df[analysis_column] = base_time.dt.normalize() + pd.to_timedelta(timestamp_values, unit="s")
        else:
            df[analysis_column] = base_time
    else:
        df[analysis_column] = base_time

    if df[analysis_column].isna().mean() > 0.2:
        raise ValueError("时间字段转换后缺失比例过高，无法进行可靠的时序分析。")

    return analysis_column


def ensure_time_features(df: pd.DataFrame, datetime_column: str) -> None:
    """确保 hour、day_of_week、day_name、month、is_weekend 等时间特征存在且一致。"""
    dt = df[datetime_column]
    df["hour"] = df["hour"] if "hour" in df.columns else dt.dt.hour
    df["day_of_week"] = df["day_of_week"] if "day_of_week" in df.columns else dt.dt.dayofweek
    df["day_name_cn"] = df["day_of_week"].map(DAY_NAME_MAP)
    df["month"] = df["month"] if "month" in df.columns else dt.dt.month
    df["date_only"] = dt.dt.date
    df["is_weekend"] = df["is_weekend"] if "is_weekend" in df.columns else df["day_of_week"].isin([5, 6]).astype(int)
    df["day_type"] = df["is_weekend"].map({0: "工作日", 1: "周末"})


def save_figure(path: Path) -> None:
    """统一保存论文可用分辨率图片。"""
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_time_series(df: pd.DataFrame, time_column: str, flow_column: str) -> Path:
    """绘制客流时间序列折线图。"""
    path = FIGURE_DIR / "01_flow_time_series.png"
    plot_df = df.sort_values(time_column)

    plt.figure(figsize=(13, 5))
    plt.plot(plot_df[time_column], plot_df[flow_column], linewidth=0.7, color="#2563eb")
    plt.title("健身房客流时间序列变化")
    plt.xlabel("时间")
    plt.ylabel("客流人数")
    plt.xticks(rotation=30)
    save_figure(path)
    return path


def plot_hourly_average(df: pd.DataFrame, flow_column: str) -> tuple[Path, pd.Series]:
    """绘制每小时平均客流柱状图。"""
    path = FIGURE_DIR / "02_hourly_average_flow.png"
    hourly_mean = df.groupby("hour")[flow_column].mean().reindex(range(24), fill_value=0)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=hourly_mean.index, y=hourly_mean.values, color="#16a34a")
    plt.title("每小时平均客流")
    plt.xlabel("小时")
    plt.ylabel("平均客流人数")
    save_figure(path)
    return path, hourly_mean


def plot_weekday_average(df: pd.DataFrame, flow_column: str) -> tuple[Path, pd.Series]:
    """绘制星期平均客流柱状图。"""
    path = FIGURE_DIR / "03_weekday_average_flow.png"
    weekday_mean = df.groupby("day_of_week")[flow_column].mean().reindex(range(7))
    weekday_mean.index = [DAY_NAME_MAP[i] for i in weekday_mean.index]

    plt.figure(figsize=(9, 5))
    sns.barplot(x=weekday_mean.index, y=weekday_mean.values, color="#f59e0b")
    plt.title("星期平均客流")
    plt.xlabel("星期")
    plt.ylabel("平均客流人数")
    save_figure(path)
    return path, weekday_mean


def plot_workday_weekend_hourly(df: pd.DataFrame, flow_column: str) -> Path:
    """绘制工作日与周末每小时客流对比折线图。"""
    path = FIGURE_DIR / "04_workday_weekend_hourly_flow.png"
    grouped = df.groupby(["hour", "day_type"])[flow_column].mean().reset_index()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=grouped, x="hour", y=flow_column, hue="day_type", marker="o")
    plt.title("工作日与周末每小时客流对比")
    plt.xlabel("小时")
    plt.ylabel("平均客流人数")
    plt.legend(title="日期类型")
    save_figure(path)
    return path


def plot_weekday_hour_heatmap(df: pd.DataFrame, flow_column: str) -> Path:
    """绘制星期-小时客流热力图。"""
    path = FIGURE_DIR / "05_weekday_hour_heatmap.png"
    pivot_df = df.pivot_table(
        values=flow_column,
        index="day_of_week",
        columns="hour",
        aggfunc="mean",
    ).reindex(index=range(7), columns=range(24))
    pivot_df.index = [DAY_NAME_MAP[i] for i in pivot_df.index]

    plt.figure(figsize=(13, 5.5))
    sns.heatmap(pivot_df, cmap="YlOrRd", linewidths=0.2, linecolor="white")
    plt.title("星期-小时平均客流热力图")
    plt.xlabel("小时")
    plt.ylabel("星期")
    save_figure(path)
    return path


def plot_flow_boxplot(df: pd.DataFrame, flow_column: str) -> tuple[Path, dict[str, float]]:
    """绘制客流箱线图并计算异常值阈值。"""
    path = FIGURE_DIR / "06_flow_boxplot.png"
    q1 = df[flow_column].quantile(0.25)
    q3 = df[flow_column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((df[flow_column] < lower_bound) | (df[flow_column] > upper_bound)).sum())

    plt.figure(figsize=(8, 5))
    sns.boxplot(y=df[flow_column], color="#a855f7")
    plt.title("客流人数箱线图")
    plt.xlabel("客流分布")
    plt.ylabel("客流人数")
    save_figure(path)

    return path, {
        "q1": float(q1),
        "q3": float(q3),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": float(outlier_count),
    }


def plot_temperature_scatter(df: pd.DataFrame, flow_column: str) -> Path | None:
    """如果存在 temperature 字段，绘制温度与客流散点图。"""
    if "temperature" not in df.columns or not is_numeric_dtype(df["temperature"]):
        return None

    path = FIGURE_DIR / "07_temperature_flow_scatter.png"

    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=df, x="temperature", y=flow_column, alpha=0.35, s=16, color="#0891b2")
    plt.title("温度与客流人数关系")
    plt.xlabel("温度")
    plt.ylabel("客流人数")
    save_figure(path)
    return path


def build_summary(
    df: pd.DataFrame,
    flow_column: str,
    figure_paths: list[Path],
    hourly_mean: pd.Series,
    weekday_mean: pd.Series,
    box_stats: dict[str, float],
    temperature_path: Path | None,
) -> str:
    """生成 EDA 主要结论文本。"""
    peak_hour = int(hourly_mean.idxmax())
    low_hour = int(hourly_mean.idxmin())
    peak_day = str(weekday_mean.idxmax())
    low_day = str(weekday_mean.idxmin())

    day_type_mean = df.groupby("day_type")[flow_column].mean()
    workday_mean = day_type_mean.get("工作日", 0)
    weekend_mean = day_type_mean.get("周末", 0)

    summary_lines = [
        "健身房客流探索性分析总结",
        "=" * 40,
        f"分析样本量：{len(df)} 条",
        f"客流字段：{flow_column}",
        f"客流均值：{df[flow_column].mean():.2f}",
        f"客流中位数：{df[flow_column].median():.2f}",
        f"客流最大值：{df[flow_column].max():.2f}",
        f"客流最小值：{df[flow_column].min():.2f}",
        "",
        "图表文件：",
    ]
    summary_lines.extend(f"- {path}" for path in figure_paths)
    if temperature_path is not None:
        summary_lines.append(f"- {temperature_path}")

    summary_lines.extend(
        [
            "",
            "主要结论：",
            "1. 客流时间序列折线图展示了健身房客流随时间连续波动的过程，可用于观察整体趋势、周期性和局部峰值。",
            f"2. 每小时平均客流柱状图显示，平均客流最高的小时为 {peak_hour} 点，最低的小时为 {low_hour} 点，说明客流存在明显日内节律。",
            f"3. 星期平均客流柱状图显示，平均客流最高的是 {peak_day}，最低的是 {low_day}，说明不同星期之间存在客流差异。",
            f"4. 工作日平均客流为 {workday_mean:.2f}，周末平均客流为 {weekend_mean:.2f}，工作日与周末的小时曲线可用于比较两类日期的高峰时段差异。",
            "5. 星期-小时热力图能同时展示周内和日内两个维度的客流强弱，颜色较深区域代表健身房使用高峰。",
            f"6. 客流箱线图显示，上四分位数为 {box_stats['q3']:.2f}，下四分位数为 {box_stats['q1']:.2f}，按 1.5 倍 IQR 规则识别到约 {int(box_stats['outlier_count'])} 条潜在异常高/低客流记录。",
        ]
    )

    if temperature_path is not None:
        corr = df[["temperature", flow_column]].corr().iloc[0, 1]
        summary_lines.append(
            f"7. 温度与客流散点图用于观察外部温度与健身房客流的关系，二者线性相关系数约为 {corr:.3f}。"
        )
    else:
        summary_lines.append("7. 数据中不存在可用 temperature 字段，因此未生成温度与客流散点图。")

    return "\n".join(summary_lines)


def main() -> None:
    """主流程：读取特征数据、生成图表、保存 EDA 总结。"""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"未找到输入文件：{INPUT_PATH}。请先运行 python3 src/feature_engineering.py。"
        )

    set_chinese_font()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = read_csv_with_fallback(INPUT_PATH)
    flow_column = identify_flow_column(df)
    raw_datetime_column = identify_datetime_column(df)
    time_column = prepare_analysis_time(df, raw_datetime_column)
    ensure_time_features(df, time_column)

    figure_paths = []
    figure_paths.append(plot_time_series(df, time_column, flow_column))
    hourly_path, hourly_mean = plot_hourly_average(df, flow_column)
    figure_paths.append(hourly_path)
    weekday_path, weekday_mean = plot_weekday_average(df, flow_column)
    figure_paths.append(weekday_path)
    figure_paths.append(plot_workday_weekend_hourly(df, flow_column))
    figure_paths.append(plot_weekday_hour_heatmap(df, flow_column))
    boxplot_path, box_stats = plot_flow_boxplot(df, flow_column)
    figure_paths.append(boxplot_path)
    temperature_path = plot_temperature_scatter(df, flow_column)

    summary_text = build_summary(
        df=df,
        flow_column=flow_column,
        figure_paths=figure_paths,
        hourly_mean=hourly_mean,
        weekday_mean=weekday_mean,
        box_stats=box_stats,
        temperature_path=temperature_path,
    )
    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")

    print("健身房客流探索性分析完成")
    print("=" * 40)
    print(f"输入文件：{INPUT_PATH}")
    print(f"图表目录：{FIGURE_DIR}")
    print(f"分析总结：{SUMMARY_PATH}")
    print("\n已生成图表：")
    for path in figure_paths:
        print(f"- {path}")
    if temperature_path is not None:
        print(f"- {temperature_path}")


if __name__ == "__main__":
    main()
