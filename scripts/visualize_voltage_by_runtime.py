"""
機台ごとの要求電圧トレンド確認ツール
  X: 累積運転時間 (h)
  Y: daily_max (kV)  + 交換タイミング縦線（exchange events CSV から）
  Y2: hours_at_31kv (h/day) バー
  ドロップダウンで機台切替
"""
from __future__ import annotations
import argparse
import pandas as pd
import plotly.graph_objects as go
import plotly.colors

DATASETS = {
    "EP370G": ("outputs/dataset_ep370g_ts_v7.csv", "outputs/exchange_events_ep370g_v5.csv"),
    "EP400G": ("outputs/dataset_ep400g_ts_v2.csv", "outputs/exchange_events_ep400g_v1.csv"),
}
THRESHOLD_CROSS = 32.0
PALETTE = plotly.colors.qualitative.G10  # 10色サイクル


def packed_to_hours(s: pd.Series) -> pd.Series:
    """累積運転時間 packed 形式 → 時間 (float)。raw = hours * 256 + minutes"""
    raw = s.fillna(0).astype(int)
    return (raw // 256) + (raw % 256) / 60


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["EP370G", "EP400G", "both"], default="both")
args = parser.parse_args()
targets = ["EP370G", "EP400G"] if args.model == "both" else [args.model]

for model in targets:
    data_path, events_path = DATASETS[model]
    df = pd.read_csv(data_path, encoding="utf-8-sig", parse_dates=["date"])
    events_df = pd.read_csv(events_path, encoding="utf-8-sig", parse_dates=["exchange_date"])
    events_df["target_no"] = events_df["target_no"].astype(str)

    id_col = "管理No"
    cum_col = "累積運転時間"
    df[cum_col] = packed_to_hours(df[cum_col])
    df["operating_hours_since_exchange"] = packed_to_hours(df["operating_hours_since_exchange"])
    df[id_col] = df[id_col].astype(str)
    machines = sorted(df[id_col].unique(), key=int)

    fig = go.Figure()
    TRACES_PER = 3  # scatter + bar + vlines

    for i, machine in enumerate(machines):
        mdf = df[df[id_col] == machine].sort_values(cum_col).copy()
        visible = i == 0

        gen_colors = [PALETTE[int(g) % len(PALETTE)] for g in mdf["gen_no"]]

        # ----- trace 0: daily_max scatter -----
        fig.add_trace(go.Scattergl(
            x=mdf[cum_col],
            y=mdf["daily_max"],
            mode="lines+markers",
            line=dict(color="rgba(150,150,150,0.5)", width=1),
            marker=dict(color=gen_colors, size=5),
            customdata=mdf[["date", "gen_no", "hours_at_31kv"]].values,
            hovertemplate=(
                "累積運転時間: %{x:,.0f} h<br>"
                "daily_max: %{y} kV<br>"
                "日付: %{customdata[0]|%Y-%m-%d}<br>"
                "gen_no: %{customdata[1]}<br>"
                "hours@31kV: %{customdata[2]:.1f} h"
                "<extra></extra>"
            ),
            name=str(machine),
            visible=visible,
            showlegend=True,
            legendgroup=f"m{machine}",
        ))

        # ----- trace 1: hours_at_31kv bars (y2) -----
        fig.add_trace(go.Bar(
            x=mdf[cum_col],
            y=mdf["hours_at_31kv"],
            marker_color="rgba(80,130,255,0.35)",
            yaxis="y2",
            name=f"{machine} h@31kV",
            visible=visible,
            showlegend=False,
            legendgroup=f"m{machine}",
            hoverinfo="skip",
        ))

        # ----- trace 2: 交換縦線 (exchange events CSV の日付 → 対応する累積運転時間) -----
        ev_dates = events_df[events_df["target_no"] == machine]["exchange_date"].sort_values()
        x_vlines: list = []
        y_vlines: list = []
        for ev_date in ev_dates:
            # イベント日付以降の最初の行の累積運転時間を取得
            row = mdf[mdf["date"] >= ev_date]
            if row.empty:
                continue
            bx = row[cum_col].iloc[0]
            x_vlines += [bx, bx, None]
            y_vlines += [13, 36, None]

        fig.add_trace(go.Scatter(
            x=x_vlines or [None],
            y=y_vlines or [None],
            mode="lines",
            line=dict(color="rgba(220,50,50,0.7)", width=1.5, dash="dash"),
            name=f"{machine} exchange",
            visible=visible,
            showlegend=False,
            hoverinfo="skip",
            legendgroup=f"m{machine}",
        ))

    # ---- dropdown buttons ----
    n = len(machines)
    buttons = []
    for i, machine in enumerate(machines):
        vis = [False] * (n * TRACES_PER)
        for j in range(TRACES_PER):
            vis[i * TRACES_PER + j] = True
        buttons.append(dict(
            label=str(machine),
            method="update",
            args=[
                {"visible": vis},
                {"title": dict(text=f"{model} - {machine}")},
            ],
        ))

    fig.update_layout(
        title=dict(text=f"{model} - {machines[0]}", x=0.5),
        xaxis=dict(title="累積運転時間 (h)", showgrid=True),
        yaxis=dict(title="daily_max (kV)", range=[13, 36]),
        yaxis2=dict(
            title="hours_at_31kV (h/day)",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, 24],
        ),
        shapes=[
            dict(
                type="line", xref="paper", x0=0, x1=1,
                yref="y", y0=THRESHOLD_CROSS, y1=THRESHOLD_CROSS,
                line=dict(color="red", dash="dot", width=1.5),
            )
        ],
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=0.01, xanchor="left",
            y=1.18, yanchor="top",
            showactive=True,
            bgcolor="white",
            bordercolor="#aaa",
        )],
        legend=dict(orientation="h", y=-0.12, x=0),
        height=550,
        margin=dict(t=100, r=80),
        hovermode="x unified",
        barmode="overlay",
    )

    out = f"outputs/voltage_by_runtime_{model}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Saved: {out}")
