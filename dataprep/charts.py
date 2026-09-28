"""Plotly visualizations over the complete loaded dataset."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .profile import correlation, numeric_values, resolve_roles

COLORS = ["#087f8c", "#ef8a47", "#5562b5", "#b54f79"]


def style(fig, title):
    fig.update_layout(
        template="plotly_white",
        title=title,
        colorway=COLORS,
        font={"family": "Arial", "color": "#172b4d"},
        margin={"l": 35, "r": 25, "t": 65, "b": 40},
        height=370,
        paper_bgcolor="white",
        legend_title_text="",
    )
    return fig


def missing_chart(frame):
    counts = pd.Series({c: int(frame[c].isna().sum()) for c in frame}).sort_values(ascending=False)
    rates = counts / max(len(frame), 1) * 100
    return style(
        go.Figure(
            go.Bar(
                x=rates.index,
                y=rates.values,
                customdata=counts.values,
                hovertemplate="%{x}<br>%{y:.2f}% thiếu (%{customdata} ô)<extra></extra>",
                marker_color=COLORS[0],
            )
        ),
        "Tỷ lệ thiếu theo cột (%)",
    )


def distribution(frame, column, role):
    s = frame[column]
    if role == "numeric":
        values = numeric_values(s).dropna().to_numpy(dtype=float)
        if not len(values):
            return style(go.Figure(), f"{column} · không có số hợp lệ")
        # Bound the bin count before allocation: automatic FD bins can explode
        # for a narrow distribution with one distant outlier.
        bins = min(60, max(1, int(np.ceil(np.sqrt(len(values))))))
        counts, edges = np.histogram(values, bins=bins)
        return style(
            go.Figure(
                go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges), marker_color=COLORS[0])
            ),
            f"{column} · phân phối trên toàn bộ dữ liệu hợp lệ",
        )
    if role == "datetime" and pd.api.types.is_datetime64_any_dtype(s):
        counts = s.dropna().dt.floor("D").value_counts().sort_index()
        return style(
            go.Figure(go.Scatter(x=counts.index, y=counts.values, mode="lines+markers")),
            f"{column} · số bản ghi theo ngày",
        )
    counts = s.dropna().astype(str).value_counts().sort_values()
    return style(
        go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=COLORS[0])),
        f"{column} · toàn bộ {len(counts):,} giá trị phân biệt",
    )


def box_chart(frame, column):
    values = numeric_values(frame[column]).dropna()
    # Precomputed quartiles keep the report compact and use the full population.
    if values.empty:
        return style(go.Figure(), column)
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    lower = values[values >= q1 - 1.5 * (q3 - q1)].min()
    upper = values[values <= q3 + 1.5 * (q3 - q1)].max()
    fig = go.Figure(
        go.Box(
            q1=[q1],
            median=[median],
            q3=[q3],
            lowerfence=[lower],
            upperfence=[upper],
            name=column,
            boxpoints=False,
        )
    )
    return style(fig, f"{column} · boxplot (không hiển thị từng điểm ngoại lệ)")


def correlation_chart(frame, roles=None, method="pearson"):
    matrix = correlation(frame, roles, method)
    if len(matrix) < 2:
        return None
    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            hovertemplate="%{x} / %{y}: %{z:.3f}<extra></extra>",
        )
    )
    return style(
        fig,
        f"Tương quan {method} · toàn bộ {matrix.attrs['total_rows']:,} dòng "
        f"× {len(matrix):,} cột số · ≥3 cặp hợp lệ",
    )


def scatter_chart(frame, x, y):
    values = pd.DataFrame({x: numeric_values(frame[x]), y: numeric_values(frame[y])}).dropna()
    fig = px.scatter(values, x=x, y=y, render_mode="webgl", opacity=0.45)
    return style(fig, f"{x} × {y} · toàn bộ {len(values):,} cặp hợp lệ · WebGL")


def comparison_chart(before, after, column):
    a = numeric_values(before[column]).dropna().to_numpy(dtype=float)
    b = numeric_values(after[column]).dropna().to_numpy(dtype=float)
    joined = np.concatenate([a, b])
    if not len(joined):
        return None
    edges = np.histogram_bin_edges(joined, bins=30)
    fig = go.Figure()
    for label, values, color in [("Trước", a, COLORS[0]), ("Sau", b, COLORS[1])]:
        counts, _ = np.histogram(values, bins=edges)
        fig.add_trace(
            go.Bar(
                x=(edges[:-1] + edges[1:]) / 2,
                y=counts,
                width=np.diff(edges),
                name=label,
                opacity=0.6,
                marker_color=color,
            )
        )
    fig.update_layout(barmode="overlay")
    return style(fig, f"{column} · trước/sau, cùng khoảng chia · số bản ghi")


def report_charts(frame, roles=None):
    resolved = resolve_roles(frame, roles)
    figures = [missing_chart(frame)]
    corr = correlation_chart(frame, resolved)
    if corr is not None:
        figures.append(corr)
    selected = [c for c in frame if resolved[c] not in ("id", "ignore")]
    figures.extend(distribution(frame, c, resolved[c]) for c in selected)
    return figures
