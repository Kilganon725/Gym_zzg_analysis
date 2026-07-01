"""
健身房客流数据清洗脚本。

从 data/raw/ 目录自动读取第一个 CSV 文件，完成时间字段转换、重复值删除、
缺失值填充、客流字段异常处理和时间排序，并将清洗结果保存到
data/processed/gym_flow_cleaned.csv。
"""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "tables"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "gym_flow_cleaned.csv"
LOG_PATH = REPORT_DIR / "preprocess_log.txt"


def find_first_csv(data_dir: Path) -> Path:
    """查找原始数据目录下按文件名排序后的第一个 CSV 文件。"""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未在 {data_dir} 目录下找到 CSV 文件。")
    return csv_files[0]


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


def detect_and_convert_datetime(df: pd.DataFrame) -> list[str]:
    """
    自动识别并转换时间字段。

    优先根据字段名中的 date、time、datetime、timestamp、日期、时间判断。
    对数值型 timestamp 做量级检查，避免把“当天秒数”误转为 1970 年时间。
    """
    time_keywords = ("date", "time", "datetime", "timestamp", "日期", "时间")
    converted_columns = []

    for column in df.columns:
        column_name = str(column).lower()
        if not any(keyword in column_name for keyword in time_keywords):
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
            converted_columns.append(column)

    return converted_columns


def identify_flow_column(df: pd.DataFrame) -> str | None:
    """尽量根据字段名识别客流人数列。"""
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

    return None


def fill_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """数值字段用中位数填充，类别字段用众数填充。"""
    missing_before = df.isna().sum()

    for column in df.columns:
        if df[column].isna().sum() == 0:
            continue

        if is_numeric_dtype(df[column]):
            fill_value = df[column].median()
            df[column] = df[column].fillna(fill_value)
        else:
            mode_values = df[column].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "未知"
            df[column] = df[column].fillna(fill_value)

    missing_after = df.isna().sum()
    return {
        "before_total": int(missing_before.sum()),
        "after_total": int(missing_after.sum()),
    }


def remove_negative_flow(df: pd.DataFrame, flow_column: str | None) -> tuple[pd.DataFrame, int]:
    """删除客流字段中的异常负数记录。"""
    if flow_column is None:
        return df, 0

    negative_mask = df[flow_column] < 0
    negative_count = int(negative_mask.sum())
    if negative_count == 0:
        return df, 0

    return df.loc[~negative_mask].copy(), negative_count


def build_log_text(
    csv_path: Path,
    original_shape: tuple[int, int],
    cleaned_shape: tuple[int, int],
    datetime_columns: list[str],
    duplicate_count: int,
    missing_info: dict[str, int],
    flow_column: str | None,
    negative_flow_count: int,
) -> str:
    """生成清洗日志文本，便于论文和实验记录引用。"""
    return "\n".join(
        [
            "健身房客流数据清洗日志",
            "=" * 40,
            f"输入文件：{csv_path}",
            f"输出文件：{CLEANED_DATA_PATH}",
            "",
            "数据量变化：",
            f"清洗前：{original_shape[0]} 行，{original_shape[1]} 列",
            f"清洗后：{cleaned_shape[0]} 行，{cleaned_shape[1]} 列",
            f"减少行数：{original_shape[0] - cleaned_shape[0]}",
            "",
            "处理步骤：",
            f"1. 时间字段识别与转换：{', '.join(datetime_columns) if datetime_columns else '未识别到可转换时间字段'}",
            f"2. 删除重复记录数量：{duplicate_count}",
            f"3. 缺失值填充前总数：{missing_info['before_total']}",
            f"4. 缺失值填充后总数：{missing_info['after_total']}",
            f"5. 客流字段识别结果：{flow_column if flow_column else '未识别到客流字段'}",
            f"6. 删除异常负数客流记录数量：{negative_flow_count}",
            f"7. 时间排序依据：{datetime_columns[0] if datetime_columns else '未执行时间排序'}",
            "",
            "清洗后字段类型：",
        ]
    )


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """组织完整清洗流程并返回清洗后的数据和日志信息。"""
    original_shape = df.shape

    datetime_columns = detect_and_convert_datetime(df)

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    missing_info = fill_missing_values(df)

    flow_column = identify_flow_column(df)
    df, negative_flow_count = remove_negative_flow(df, flow_column)

    if datetime_columns:
        df = df.sort_values(by=datetime_columns[0]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    log_info = {
        "original_shape": original_shape,
        "cleaned_shape": df.shape,
        "datetime_columns": datetime_columns,
        "duplicate_count": duplicate_count,
        "missing_info": missing_info,
        "flow_column": flow_column,
        "negative_flow_count": negative_flow_count,
    }
    return df, log_info


def main() -> None:
    """主函数：读取原始数据、执行清洗、保存结果和日志。"""
    csv_path = find_first_csv(RAW_DATA_DIR)
    df = read_csv_with_fallback(csv_path)

    cleaned_df, log_info = preprocess_data(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False, encoding="utf-8-sig")

    log_text = build_log_text(csv_path=csv_path, **log_info)
    log_text = f"{log_text}\n{cleaned_df.dtypes.to_string()}\n"
    LOG_PATH.write_text(log_text, encoding="utf-8")

    print(log_text)
    print(f"清洗后数据已保存至：{CLEANED_DATA_PATH}")
    print(f"清洗日志已保存至：{LOG_PATH}")


if __name__ == "__main__":
    main()
