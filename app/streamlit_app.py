"""
健身房客流分析可视化系统。

用于毕业设计《基于 Python 的健身房客流时序数据挖掘与变化特征分析》的
答辩展示，包含数据概览、客流趋势、客流特征、预测结果和运营建议页面。
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.api.types import is_numeric_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "gym_flow_features.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "tables" / "model_metrics.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
PREDICTION_COMPARE_PATH = FIGURE_DIR / "prediction_compare.png"
FEATURE_IMPORTANCE_PATH = FIGURE_DIR / "feature_importance.png"

DAY_NAME_MAP = {
    0: "周一",
    1: "周二",
    2: "周三",
    3: "周四",
    4: "周五",
    5: "周六",
    6: "周日",
}


st.set_page_config(
    page_title="健身房客流分析系统",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """读取特征数据并构造展示所需字段。"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件：{DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    flow_column = identify_flow_column(df)
    df["analysis_datetime"] = build_analysis_datetime(df)
    df = df.sort_values("analysis_datetime").reset_index(drop=True)

    df["hour"] = df["analysis_datetime"].dt.hour
    df["day_of_week"] = df["analysis_datetime"].dt.dayofweek
    df["day_name_cn"] = df["day_of_week"].map(DAY_NAME_MAP)
    df["month"] = df["analysis_datetime"].dt.month
    df["date_only"] = df["analysis_datetime"].dt.date
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_type"] = df["is_weekend"].map({0: "工作日", 1: "周末"})
    df.attrs["flow_column"] = flow_column
    return df


@st.cache_data
def load_metrics() -> pd.DataFrame:
    """读取模型评价结果。"""
    if not METRICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(METRICS_PATH, encoding="utf-8-sig")


def identify_flow_column(df: pd.DataFrame) -> str:
    """自动识别客流人数字段。"""
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

    raise ValueError("未能识别客流人数字段，请检查数据字段。")


def build_analysis_datetime(df: pd.DataFrame) -> pd.Series:
    """组合 date 和 timestamp，得到完整时间轴。"""
    if "date" not in df.columns:
        raise ValueError("数据中缺少 date 字段，无法构造时间轴。")

    base_time = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    if base_time.notna().mean() < 0.8:
        raise ValueError("date 字段无法可靠转换为时间类型。")

    date_only = (
        base_time.dt.hour.eq(0)
        & base_time.dt.minute.eq(0)
        & base_time.dt.second.eq(0)
    ).all()
    if date_only and "timestamp" in df.columns and is_numeric_dtype(df["timestamp"]):
        seconds = pd.to_numeric(df["timestamp"], errors="coerce")
        if seconds.between(0, 86_399).mean() >= 0.8:
            return base_time.dt.normalize() + pd.to_timedelta(seconds, unit="s")

    return base_time


def page_header(title: str, subtitle: str) -> None:
    """统一页面标题。"""
    st.title(title)
    st.caption(subtitle)


def show_data_overview(df: pd.DataFrame, flow_column: str) -> None:
    """数据概览页面。"""
    page_header("数据概览", "查看样本规模、字段结构、时间范围与原始数据样例")

    min_time = df["analysis_datetime"].min()
    max_time = df["analysis_datetime"].max()
    metric_cols = st.columns(4)
    metric_cols[0].metric("数据量", f"{len(df):,} 条")
    metric_cols[1].metric("字段数", f"{df.shape[1]} 个")
    metric_cols[2].metric("平均客流", f"{df[flow_column].mean():.2f}")
    metric_cols[3].metric("最大客流", f"{df[flow_column].max():.0f}")

    st.subheader("时间范围")
    st.write(f"{min_time:%Y-%m-%d %H:%M:%S} 至 {max_time:%Y-%m-%d %H:%M:%S}")

    st.subheader("字段信息")
    field_info = pd.DataFrame(
        {
            "字段名": df.columns,
            "字段类型": [str(dtype) for dtype in df.dtypes],
            "缺失值数量": df.isna().sum().values,
        }
    )
    st.dataframe(field_info, width="stretch", hide_index=True)

    st.subheader("前 10 行数据")
    st.dataframe(df.head(10), width="stretch")


def show_flow_trend(df: pd.DataFrame, flow_column: str) -> None:
    """客流趋势页面。"""
    page_header("客流趋势", "展示客流随时间变化的趋势，并支持按小时、星期、月份查看平均客流")

    date_range = st.slider(
        "选择时间范围",
        min_value=df["analysis_datetime"].dt.date.min(),
        max_value=df["analysis_datetime"].dt.date.max(),
        value=(df["analysis_datetime"].dt.date.min(), df["analysis_datetime"].dt.date.max()),
    )
    filtered_df = df[
        (df["analysis_datetime"].dt.date >= date_range[0])
        & (df["analysis_datetime"].dt.date <= date_range[1])
    ]

    fig = px.line(
        filtered_df,
        x="analysis_datetime",
        y=flow_column,
        title="客流时间序列变化",
        labels={"analysis_datetime": "时间", flow_column: "客流人数"},
    )
    fig.update_traces(line=dict(width=1))
    st.plotly_chart(fig, width="stretch")

    view_type = st.segmented_control(
        "平均客流查看方式",
        options=["按小时", "按星期", "按月份"],
        default="按小时",
    )

    if view_type == "按小时":
        group_df = filtered_df.groupby("hour", as_index=False)[flow_column].mean()
        fig_bar = px.bar(
            group_df,
            x="hour",
            y=flow_column,
            title="每小时平均客流",
            labels={"hour": "小时", flow_column: "平均客流人数"},
        )
    elif view_type == "按星期":
        group_df = (
            filtered_df.groupby(["day_of_week", "day_name_cn"], as_index=False)[flow_column]
            .mean()
            .sort_values("day_of_week")
        )
        fig_bar = px.bar(
            group_df,
            x="day_name_cn",
            y=flow_column,
            title="星期平均客流",
            labels={"day_name_cn": "星期", flow_column: "平均客流人数"},
        )
    else:
        group_df = filtered_df.groupby("month", as_index=False)[flow_column].mean()
        fig_bar = px.bar(
            group_df,
            x="month",
            y=flow_column,
            title="月份平均客流",
            labels={"month": "月份", flow_column: "平均客流人数"},
        )

    st.plotly_chart(fig_bar, width="stretch")


def show_flow_features(df: pd.DataFrame, flow_column: str) -> None:
    """客流特征页面。"""
    page_header("客流特征", "对比工作日与周末差异，观察星期和小时维度下的客流结构")

    grouped = df.groupby(["hour", "day_type"], as_index=False)[flow_column].mean()
    fig_line = px.line(
        grouped,
        x="hour",
        y=flow_column,
        color="day_type",
        markers=True,
        title="工作日与周末每小时客流对比",
        labels={"hour": "小时", flow_column: "平均客流人数", "day_type": "日期类型"},
    )
    st.plotly_chart(fig_line, width="stretch")

    heatmap_df = (
        df.pivot_table(values=flow_column, index="day_of_week", columns="hour", aggfunc="mean")
        .reindex(index=range(7), columns=range(24))
    )
    heatmap_df.index = [DAY_NAME_MAP[i] for i in heatmap_df.index]
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=list(heatmap_df.columns),
            y=list(heatmap_df.index),
            colorscale="YlOrRd",
            colorbar=dict(title="平均客流"),
        )
    )
    fig_heatmap.update_layout(
        title="星期-小时平均客流热力图",
        xaxis_title="小时",
        yaxis_title="星期",
        height=520,
    )
    st.plotly_chart(fig_heatmap, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        fig_box = px.box(
            df,
            y=flow_column,
            title="客流箱线图",
            labels={flow_column: "客流人数"},
        )
        st.plotly_chart(fig_box, width="stretch")
    with col2:
        if "temperature" in df.columns:
            fig_scatter = px.scatter(
                df,
                x="temperature",
                y=flow_column,
                opacity=0.35,
                title="温度与客流关系",
                labels={"temperature": "温度", flow_column: "客流人数"},
            )
            st.plotly_chart(fig_scatter, width="stretch")
        else:
            st.info("当前数据中不存在 temperature 字段，未展示温度散点图。")


def show_prediction_results() -> None:
    """预测结果页面。"""
    page_header("预测结果", "展示模型评价指标、真实值与预测值对比、特征重要性")

    metrics_df = load_metrics()
    if metrics_df.empty:
        st.warning("未找到模型评价文件，请先运行 python3 src/model_train.py。")
        return

    st.subheader("模型评价结果")
    st.dataframe(
        metrics_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE": "{:.4f}", "R2": "{:.4f}"}),
        width="stretch",
    )

    best_model = metrics_df.sort_values("RMSE").iloc[0]
    metric_cols = st.columns(4)
    metric_cols[0].metric("最佳模型", str(best_model["model"]))
    metric_cols[1].metric("MAE", f"{best_model['MAE']:.3f}")
    metric_cols[2].metric("RMSE", f"{best_model['RMSE']:.3f}")
    metric_cols[3].metric("R2", f"{best_model['R2']:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("真实值与预测值对比")
        if PREDICTION_COMPARE_PATH.exists():
            st.image(str(PREDICTION_COMPARE_PATH), width="stretch")
        else:
            st.warning("未找到 prediction_compare.png。")
    with col2:
        st.subheader("特征重要性")
        if FEATURE_IMPORTANCE_PATH.exists():
            st.image(str(FEATURE_IMPORTANCE_PATH), width="stretch")
        else:
            st.warning("未找到 feature_importance.png。")


def show_operation_advice(df: pd.DataFrame, flow_column: str) -> None:
    """运营建议页面。"""
    page_header("运营建议", "基于探索性分析结果自动生成高峰、低谷和管理建议")

    hourly_mean = df.groupby("hour")[flow_column].mean().sort_values(ascending=False)
    weekday_hour = (
        df.groupby(["day_of_week", "day_name_cn", "hour"], as_index=False)[flow_column]
        .mean()
        .sort_values(flow_column, ascending=False)
    )
    day_type_mean = df.groupby("day_type")[flow_column].mean()

    peak_hours = hourly_mean.head(3)
    low_hours = hourly_mean.tail(3).sort_values()
    peak_slots = weekday_hour.head(5)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("高峰时段")
        st.dataframe(
            peak_hours.rename("平均客流").reset_index().rename(columns={"hour": "小时"}),
            width="stretch",
            hide_index=True,
        )
    with col2:
        st.subheader("低谷时段")
        st.dataframe(
            low_hours.rename("平均客流").reset_index().rename(columns={"hour": "小时"}),
            width="stretch",
            hide_index=True,
        )

    st.subheader("星期-小时高峰组合")
    st.dataframe(
        peak_slots.rename(columns={"day_name_cn": "星期", "hour": "小时", flow_column: "平均客流"})[
            ["星期", "小时", "平均客流"]
        ],
        width="stretch",
        hide_index=True,
    )

    workday_mean = day_type_mean.get("工作日", 0)
    weekend_mean = day_type_mean.get("周末", 0)
    highest_hour = int(peak_hours.index[0])
    lowest_hour = int(low_hours.index[0])

    st.subheader("管理建议")
    suggestions = [
        f"在 {highest_hour} 点附近的客流高峰时段增加前台、巡场和保洁人员，减少器械等待和服务拥堵。",
        f"在 {lowest_hour} 点附近的低谷时段安排设备维护、场地清洁和私教复盘，降低对会员体验的影响。",
        "针对高峰星期-小时组合提前安排热门器械区域管理，必要时设置课程预约和分流提醒。",
        "可在低谷时段推出团课、私教体验或会员积分活动，提高场馆利用率。",
    ]
    if workday_mean > weekend_mean:
        suggestions.append("当前工作日平均客流高于周末，可在工作日晚间加强运营排班和会员引导。")
    else:
        suggestions.append("当前周末平均客流高于工作日，可在周末增加课程供给和现场服务人员。")

    for suggestion in suggestions:
        st.write(f"- {suggestion}")


def main() -> None:
    """Streamlit 主入口。"""
    st.sidebar.title("健身房客流分析系统")
    st.sidebar.caption("毕业设计可视化展示")
    page = st.sidebar.radio(
        "页面导航",
        ["数据概览", "客流趋势", "客流特征", "预测结果", "运营建议"],
    )

    try:
        df = load_data()
        flow_column = df.attrs["flow_column"]
    except Exception as exc:
        st.error(f"数据加载失败：{exc}")
        st.stop()

    if page == "数据概览":
        show_data_overview(df, flow_column)
    elif page == "客流趋势":
        show_flow_trend(df, flow_column)
    elif page == "客流特征":
        show_flow_features(df, flow_column)
    elif page == "预测结果":
        show_prediction_results()
    else:
        show_operation_advice(df, flow_column)


if __name__ == "__main__":
    main()
