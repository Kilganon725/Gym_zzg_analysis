"""
原始数据概览脚本。

用于读取 data/raw/ 目录下的第一个 CSV 文件，输出数据规模、字段信息、
缺失值、重复行和时间字段识别结果，便于毕业论文第 3 章数据说明使用。
"""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT_PATH = REPORT_DIR / "data_overview.txt"


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
    根据字段名识别潜在时间字段，并尝试转换为 datetime 类型。

    只对字段名包含 date、time、datetime、timestamp 的列进行尝试；
    转换成功率达到 80% 时，认为该字段是有效时间字段。
    """
    time_keywords = ("date", "time", "datetime", "timestamp", "日期", "时间")
    converted_columns = []

    for column in df.columns:
        column_name = str(column).lower()
        if not any(keyword in column_name for keyword in time_keywords):
            continue

        # 数值型 timestamp 可能只是“当天秒数”等编码，不一定是完整日期时间。
        # 只有达到常见 Unix 秒级时间戳量级时才自动转换，避免误转为 1970 年。
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


def build_overview_text(df: pd.DataFrame, csv_path: Path, datetime_columns: list[str]) -> str:
    """生成可打印、可保存的数据概览文本。"""
    row_count, column_count = df.shape
    missing_count = df.isna().sum()
    missing_rate = (missing_count / row_count * 100).round(2)
    missing_summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_rate_percent": missing_rate,
        }
    )
    duplicated_count = int(df.duplicated().sum())

    sections = [
        "健身房客流原始数据概览",
        "=" * 40,
        f"数据文件：{csv_path}",
        f"数据行数：{row_count}",
        f"数据列数：{column_count}",
        "",
        "字段名：",
        ", ".join(map(str, df.columns)),
        "",
        "字段类型：",
        df.dtypes.to_string(),
        "",
        "前 5 行数据：",
        df.head(5).to_string(index=False),
        "",
        "缺失值统计：",
        missing_summary.to_string(),
        "",
        f"重复行数量：{duplicated_count}",
        "",
        "时间字段识别与转换：",
        ", ".join(map(str, datetime_columns)) if datetime_columns else "未识别到可转换的时间字段",
    ]

    return "\n".join(sections)


def main() -> None:
    """主流程：读取数据、识别时间字段、打印并保存概览结果。"""
    csv_path = find_first_csv(RAW_DATA_DIR)
    df = read_csv_with_fallback(csv_path)
    datetime_columns = detect_and_convert_datetime(df)

    overview_text = build_overview_text(df, csv_path, datetime_columns)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(overview_text, encoding="utf-8")

    print(overview_text)
    print(f"\n数据概览结果已保存至：{REPORT_PATH}")


if __name__ == "__main__":
    main()
