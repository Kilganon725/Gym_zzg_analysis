"""
健身房客流时序特征工程脚本。

输入清洗后的 data/processed/gym_flow_cleaned.csv，自动识别时间字段和
客流人数字段，构造时间特征、滞后特征、滚动均值和变化率特征，并保存到
data/processed/gym_flow_features.csv。
"""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "gym_flow_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "gym_flow_features.csv"


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


def detect_datetime_column(df: pd.DataFrame) -> str:
    """
    自动识别时间字段并转换为 datetime 类型。

    优先识别字段名含 date、time、datetime、timestamp、日期、时间的列。
    对数值型 timestamp 做量级检查，避免把“当天秒数”误判为完整日期。
    """
    time_keywords = ("date", "time", "datetime", "timestamp", "日期", "时间")
    candidates = []

    for column in df.columns:
        column_name = str(column).lower()
        if not any(keyword in column_name for keyword in time_keywords):
            continue

        if is_datetime64_any_dtype(df[column]):
            candidates.append((column, 1.0))
            continue

        if is_numeric_dtype(df[column]):
            numeric_values = pd.to_numeric(df[column], errors="coerce").dropna()
            if numeric_values.empty or numeric_values.median() < 1_000_000_000:
                continue
            parsed = pd.to_datetime(df[column], errors="coerce", unit="s")
        else:
            parsed = pd.to_datetime(df[column], errors="coerce", utc=True)

        valid_rate = parsed.notna().mean()
        if valid_rate >= 0.8:
            df[column] = parsed
            candidates.append((column, valid_rate))

    if not candidates:
        raise ValueError(
            "未能识别时间字段。请确认数据中包含 date、time、datetime、timestamp、日期或时间等字段，"
            "且该字段可以被 pandas 转换为 datetime 类型。"
        )

    # 选择转换成功率最高的时间字段；若成功率相同，保留原字段顺序靠前者。
    return sorted(candidates, key=lambda item: item[1], reverse=True)[0][0]


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


def add_time_features(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """构造小时、星期、月份、日期和周末标记等时间特征。"""
    dt = df[datetime_column]

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["day_name"] = dt.dt.day_name()
    df["month"] = dt.dt.month
    df["date"] = dt.dt.date
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def add_flow_features(df: pd.DataFrame, flow_column: str) -> pd.DataFrame:
    """基于客流字段构造滞后、滚动均值和变化率特征。"""
    df["lag_1"] = df[flow_column].shift(1)
    df["lag_2"] = df[flow_column].shift(2)
    df["lag_3"] = df[flow_column].shift(3)

    df["rolling_mean_3"] = df[flow_column].rolling(window=3).mean()
    df["rolling_mean_6"] = df[flow_column].rolling(window=6).mean()
    df["rolling_mean_24"] = df[flow_column].rolling(window=24).mean()

    previous_flow = df[flow_column].shift(1).astype(float)
    current_flow = df[flow_column].astype(float)
    df["change_rate"] = (current_flow - previous_flow) / previous_flow
    df.loc[previous_flow == 0, "change_rate"] = 0
    df["change_rate"] = df["change_rate"].fillna(0)

    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """组织特征工程流程，返回特征数据、时间字段和客流字段。"""
    datetime_column = detect_datetime_column(df)
    flow_column = identify_flow_column(df)

    df = df.sort_values(by=datetime_column).reset_index(drop=True)
    df = add_time_features(df, datetime_column)
    df = add_flow_features(df, flow_column)

    df = df.dropna(
        subset=[
            "lag_1",
            "lag_2",
            "lag_3",
            "rolling_mean_3",
            "rolling_mean_6",
            "rolling_mean_24",
        ]
    ).reset_index(drop=True)

    return df, datetime_column, flow_column


def main() -> None:
    """主函数：读取清洗数据、构造特征、保存结果并打印字段列表。"""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"未找到输入文件：{INPUT_PATH}。请先运行 python3 src/data_preprocess.py。"
        )

    df = read_csv_with_fallback(INPUT_PATH)
    feature_df, datetime_column, flow_column = engineer_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("健身房客流特征工程完成")
    print("=" * 40)
    print(f"输入文件：{INPUT_PATH}")
    print(f"输出文件：{OUTPUT_PATH}")
    print(f"时间字段：{datetime_column}")
    print(f"客流字段：{flow_column}")
    print(f"最终数据量：{feature_df.shape[0]} 行，{feature_df.shape[1]} 列")
    print("\n最终特征字段列表：")
    for column in feature_df.columns:
        print(f"- {column}")


if __name__ == "__main__":
    main()
